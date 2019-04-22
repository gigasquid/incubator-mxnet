(ns bert.bert-sentence-classification
  (:require [bert.util :as bert-util]
            [clojure-csv.core :as csv]
            [clojure.string :as string]
            [org.apache.clojure-mxnet.callback :as callback]
            [org.apache.clojure-mxnet.context :as context]
            [org.apache.clojure-mxnet.dtype :as dtype]
            [org.apache.clojure-mxnet.io :as mx-io]
            [org.apache.clojure-mxnet.layout :as layout]
            [org.apache.clojure-mxnet.module :as m]
            [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.optimizer :as optimizer]
            [org.apache.clojure-mxnet.symbol :as sym]))

(def model-path-prefix "data/static_bert_base_net")
;; epoch number of the model
(def epoch 0)
;; the vocabulary used in the model
(def model-vocab "data/vocab.json")
;; the input question
;; the maximum length of the sequence
(def seq-length 128)


(defn pre-processing [ctx idx->token token->idx train-item]
    (let [[sentence-a sentence-b label] train-item
       ;;; pre-processing tokenize sentence
          token-1 (bert-util/tokenize (string/lower-case sentence-a))
          token-2 (bert-util/tokenize (string/lower-case sentence-b))
          valid-length (+ (count token-1) (count token-2))
        ;;; generate token types [0000...1111...0000]
          qa-embedded (into (bert-util/pad [] 0 (count token-1))
                            (bert-util/pad [] 1 (count token-2)))
          token-types (bert-util/pad qa-embedded 0 seq-length)
        ;;; make BERT pre-processing standard
          token-2 (conj token-2 "[SEP]")
          token-1 (into [] (concat ["[CLS]"] token-1 ["[SEP]"] token-2))
          tokens (bert-util/pad token-1 "[PAD]" seq-length)
        ;;; pre-processing - token to index translation
          indexes (bert-util/tokens->idxs token->idx tokens)]
    {:input-batch [indexes
                   token-types
                   [valid-length]]
     :label (if (= "0" label)
              [0]
              [1])
     :tokens tokens
     :train-item train-item}))

(defn fine-tune-model
  "msymbol: the pretrained network symbol
    arg-params: the argument parameters of the pretrained model
    num-classes: the number of classes for the fine-tune datasets"
  [msymbol {:keys [num-classes dropout]}]
  (as-> msymbol data
    (sym/dropout {:data data :p dropout})
    (sym/fully-connected "fc-finetune" {:data data :num-hidden num-classes})
    (sym/softmax-output "softmax" {:data data})))


(defn slice-inputs-data [processed-datas n]
  (->> processed-datas
       (mapv #(nth (:input-batch %) n))
       (flatten)
       (into [])))

(defn prepare-data [dev]
  (let [raw-file (csv/parse-csv (slurp "data/dev.tsv") :delimiter \tab)
        vocab (bert-util/get-vocab)
        idx->token (:idx->token vocab)
        token->idx (:token->idx vocab)
        data-train-raw (->> raw-file
                            (mapv #(vals (select-keys % [3 4 0])))
                            (rest) ;;drop header
                            (into []))
        processed-datas (mapv #(pre-processing dev idx->token token->idx %) data-train-raw)]
    {:data0s (slice-inputs-data processed-datas 0)
     :data1s (slice-inputs-data processed-datas 1)
     :data2s (slice-inputs-data processed-datas 2)
     :labels (->> (mapv :label processed-datas)
                  (flatten)
                  (into []))
     :train-num (count processed-datas)}))


(defn train [dev num-epoch]
  (let [bert-base (m/load-checkpoint {:prefix model-path-prefix :epoch 0})
        model-sym (fine-tune-model (m/symbol bert-base) {:num-classes 2 :dropout 0.1})
        {:keys [data0s data1s data2s labels train-num]} (prepare-data dev)
        batch-size 32
        data-desc0 (mx-io/data-desc {:name "data0"
                                   :shape [train-num seq-length]
                                   :dtype dtype/FLOAT32
                                     :layout layout/NT})
        data-desc1 (mx-io/data-desc {:name "data1"
                                   :shape [train-num seq-length]
                                   :dtype dtype/FLOAT32
                                     :layout layout/NT})
        data-desc2 (mx-io/data-desc {:name "data2"
                                     :shape [train-num]
                                     :dtype dtype/FLOAT32
                                     :layout layout/N})
        label-desc (mx-io/data-desc {:name "softmax_label"
                                     :shape [train-num]
                                   :dtype dtype/FLOAT32
                                     :layout layout/N})
        train-data  (mx-io/ndarray-iter {data-desc0 (ndarray/array data0s [train-num seq-length]
                                                                  {:ctx dev})
                                        data-desc1 (ndarray/array data1s [train-num seq-length]
                                                                  {:ctx dev})
                                        data-desc2 (ndarray/array data2s [train-num]
                                                                  {:ctx dev})}
                                       {:label {label-desc (ndarray/array labels [train-num]
                                                                           {:ctx dev})}
                                        :data-batch-size batch-size})
        model (m/module model-sym {:contexts [dev]
                                   :data-names ["data0" "data1" "data2"]})]
    (m/fit model {:train-data train-data  :num-epoch num-epoch
                  :fit-params (m/fit-params {:allow-missing true
                                             :arg-params (m/arg-params bert-base)
                                             :aux-params (m/aux-params bert-base)
                                             :optimizer (optimizer/adam {:learning-rate 5e-6 :episilon 1e-9})
                                             :batch-end-callback (callback/speedometer batch-size 1)})})))

(defn -main [& args]
  (let [[dev] args]
    (if (= dev ":gpu")
      (train (context/gpu) 3)
      (train (context/cpu) 3))))

(comment

  (train (context/cpu 0) 3)
  (m/save-checkpoint model {:prefix "fine-tune-sentence-bert" :epoch 3})

)
