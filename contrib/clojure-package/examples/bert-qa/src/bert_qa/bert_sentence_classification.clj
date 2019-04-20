(ns bert-qa.bert-sentence-classification
  (:require [clojure.string :as string]
            [clojure.reflect :as r]
            [cheshire.core :as json]
            [clojure.java.io :as io]
            [clojure.set :as set]
            [org.apache.clojure-mxnet.dtype :as dtype]
            [org.apache.clojure-mxnet.context :as context]
            [org.apache.clojure-mxnet.layout :as layout]
            [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.symbol :as sym]
            [org.apache.clojure-mxnet.module :as m]
            [org.apache.clojure-mxnet.infer :as infer]
            [org.apache.clojure-mxnet.eval-metric :as eval-metric]
            [org.apache.clojure-mxnet.optimizer :as optimizer]
            [clojure.pprint :as pprint]
            [clojure-csv.core :as csv]
            [bert-qa.infer :as bert-infer]))


(def model-path-prefix "model/bert-base")
;; epoch number of the model
(def epoch 0)
;; the vocabulary used in the model
(def model-vocab "model/vocab.json")
;; the input question
;; the maximum length of the sequence
(def seq-length 128)


(defn pre-processing [ctx idx->token token->idx train-item]
    (let [[sentence-a sentence-b label] train-item
       ;;; pre-processing tokenize sentence
          token-1 (bert-infer/tokenize (string/lower-case sentence-a))
          token-2 (bert-infer/tokenize (string/lower-case sentence-b))
          valid-length (+ (count token-1) (count token-2))
        ;;; generate token types [0000...1111...0000]
          qa-embedded (into (bert-infer/pad [] 0 (count token-1))
                            (bert-infer/pad [] 1 (count token-2)))
          token-types (bert-infer/pad qa-embedded 0 seq-length)
        ;;; make BERT pre-processing standard
          token-2 (conj token-2 "[SEP]")
          token-1 (into [] (concat ["[CLS]"] token-1 ["[SEP]"] token-2))
          tokens (bert-infer/pad token-1 "[PAD]" seq-length)
        ;;; pre-processing - token to index translation
          indexes (bert-infer/tokens->idxs token->idx tokens)]
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


(comment
  
 (do

;;; load the pre-trained BERT model using the module api
  
   (def bert-base (m/load-checkpoint {:prefix model-path-prefix :epoch 0}))
;;; now that we have loaded the BERT model we need to attach an additional layer for classification which is a dense layer with 2 classes
   (def model-sym (fine-tune-model (m/symbol bert-base) {:num-classes 2 :dropout 0.1}))
   (def arg-params (m/arg-params bert-base))
   (def aux-params (m/aux-params bert-base))

   (def devs [(context/default-context)])
   (def input-descs [{:name "data0"
                      :shape [1 seq-length]
                      :dtype dtype/FLOAT32
                      :layout layout/NT}
                     {:name "data1"
                      :shape [1 seq-length]
                      :dtype dtype/FLOAT32
                      :layout layout/NT}
                     {:name "data2"
                      :shape [1]
                      :dtype dtype/FLOAT32
                      :layout layout/N}])
   (def label-descs [{:name "softmax_label"
                      :shape [1 2]
                      :dtype dtype/FLOAT32
                      :layout layout/NT}])

;;; Data Preprocessing for BERT

   ;; For demonstration purpose, we use the dev set of the Microsoft Research Paraphrase Corpus dataset. The file is named ‘dev.tsv’. Let’s take a look at the raw dataset.
   ;; it contains 5 columns seperated by tabs
   (def raw-file (->> (string/split (slurp "dev.tsv") #"\n")
                      (map #(string/split % #"\t") )))
   (def raw-file (csv/parse-csv (slurp "dev.tsv") :delimiter \tab))
   (take 3 raw-file)
   ;; (["﻿Quality" "#1 ID" "#2 ID" "#1 String" "#2 String"]
   ;; ["1"
   ;;  "1355540"
   ;;  "1355592"
   ;;  "He said the foodservice pie business doesn 't fit the company 's long-term growth strategy ."
   ;;  "\" The foodservice pie business does not fit our long-term growth strategy ."]
   ;; ["0"
   ;;  "2029631"
   ;;  "2029565"
   ;;  "Magnarelli said Racicot hated the Iraqi regime and looked forward to using his long years of training in the war ."
   ;;  "His wife said he was \" 100 percent behind George Bush \" and looked forward to using his years of training in the war ."])

;;; for our task we are only interested in the 0 3rd and 4th column
   (vals (select-keys (first raw-file) [3 4 0]))
                                        ;=> ("#1 String" "#2 String" "﻿Quality")
   (def data-train-raw (->> raw-file
                            (mapv #(vals (select-keys % [3 4 0])))
                            (rest) ;;drop header
                            (into [])
                            ))
   (def sample (first data-train-raw))
   (nth sample 0) ;;;sentence a
                                        ;=> "He said the foodservice pie business doesn 't fit the company 's long-term growth strategy ."
   (nth sample 1)                       ;; sentence b
   "\" The foodservice pie business does not fit our long-term growth strategy ."

   (nth sample 2)         ; 1 means equivalent, 0 means not equivalent
                                        ;=> "1"

;;; Now we need to turn these into ndarrays to make a Data Iterator
   (def vocab (bert-infer/get-vocab))
   (def idx->token (:idx->token vocab))
   (def token->idx (:token->idx vocab))

  

;;; our sample item
   (def sample-data (pre-processing (context/default-context) idx->token token->idx sample))

   (def train-count (count data-train-raw)) ;=> 389

   ;; now create the module

   (def lr 5e-6)
   
   (def model (-> (m/module model-sym {:contexts devs
                                       :data-names ["data0" "data1" "data2"]})
                  (m/bind {:data-shapes input-descs :label-shapes label-descs})
                  (m/init-params {:arg-params arg-params :aux-params aux-params
                                  :allow-missing true})
                  (m/init-optimizer {:optimizer (optimizer/adam {:learning-rate lr :episilon 1e-9})})))

   (def metric (eval-metric/accuracy))
   (def num-epoch 1)
   (def processed-datas (mapv #(pre-processing (context/default-context) idx->token token->idx %)
                              data-train-raw))
   (def batch-size 32)

   )



 (def total-number (count processed-datas)) ;=> 389
 (def train-num (int (* 0.8 total-number)))
 (def train-eval-sets (partition-all train-num processed-datas))
 (map count train-eval-sets) ;=> (311 78)
 (def train-processed-datas (first train-eval-sets))
 (def eval-processed-datas (second train-eval-sets))
 (def train-num (count train-processed-datas))
 (def eval-num (count eval-processed-datas))

 ;;; to do split up into training/ eval

  (def data0s (->> (mapv #(nth (:input-batch %) 0) processed-datas)
                  (flatten)
                  (into [])))
 (def data1s (->> (mapv #(nth (:input-batch %) 1) processed-datas)
                  (flatten)
                  (into [])))
 (def data2s (->> (mapv #(nth (:input-batch %) 2) processed-datas)
                  (flatten)
                  (into [])))
 (def labels (->> (mapv :label processed-datas)
                  (flatten)
                  (into [])))




 (def data-desc0 (mx-io/data-desc {:name "data0"
                                   :shape [train-num seq-length]
                                   :dtype dtype/FLOAT32
                                   :layout layout/NT}))

 (def data-desc1 (mx-io/data-desc {:name "data1"
                                   :shape [train-num seq-length]
                                   :dtype dtype/FLOAT32
                                   :layout layout/NT}))
 (def data-desc2 (mx-io/data-desc {:name "data2"
                                   :shape [train-num]
                                   :dtype dtype/FLOAT32
                                   :layout layout/N}))
 (def label-desc (mx-io/data-desc {:name "softmax_label"
                                   :shape [train-num]
                                   :dtype dtype/FLOAT32
                                   :layout layout/N}))

 (def train-data (mx-io/ndarray-iter {data-desc0 (ndarray/array data0s [train-num seq-length])
                                      data-desc1 (ndarray/array data1s [train-num seq-length])
                                      data-desc2 (ndarray/array data2s [train-num])}
                                     {:label {label-desc (ndarray/array labels [train-num])}
                                      :data-batch-size batch-size
                                      :last-batch-handle "pad"}))
 (mx-io/provide-data-desc train-data)
 (mx-io/provide-label-desc train-data)

 (def batch (mx-io/next train-data))
 (mx-io/batch-data batch)
 (mx-io/batch-label batch)

    (def model (-> (m/module model-sym {:contexts devs
                                       :data-names ["data0" "data1" "data2"]})
                   (m/bind {:data-shapes (mx-io/provide-data-desc train-data)
                            :label-shapes (mx-io/provide-label-desc train-data)})
                  (m/init-params {:arg-params arg-params :aux-params aux-params
                                  :allow-missing true})
                  (m/init-optimizer {:optimizer (optimizer/adam {:learning-rate lr :episilon 1e-9})})))

 (-> model
     (m/forward {:data (mx-io/batch-data batch)})
     (m/backward)
     (m/update))

 (m/fit model {:train-data train-data  :num-epoch num-epoch
               :fit-params (m/fit-params {:allow-missing true
                                          :arg-params arg-params :aux-params aux-params
                                          :optimizer (optimizer/adam {:learning-rate lr :episilon 1e-9})
                                          :batch-end-callback (callback/speedometer batch-size batch-size)})})



 #_(m/save-checkpoint model {:prefix "fine-tune-sentence-bert" :epoch 0 :save-opt-states true})

 (def clojure-test-data (pre-processing (context/default-context) idx->token token->idx
                                        ["Rich Hickey is the creator of the Clojure language."
                                         "The Clojure language was Rich Hickey." "1"]))

 
(-> model
    (m/forward {:data (:input-batch sample-data)})
    (m/outputs)
    (ffirst)
    (ndarray/->vec)
    (zipmap [:equivalent :not-equivalent]))

(-> model
    (m/forward {:data (:input-batch clojure-test-data)})
    (m/outputs)
    (ffirst)
    (ndarray/->vec)
    (zipmap [:equivalent :not-equivalent]))) ()
