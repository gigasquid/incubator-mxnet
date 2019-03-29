(ns org.apache.clojure-mxnet.bert
  (:require [clojure.java.io :as io]
            [clojure.java.shell :refer [sh]]
            [clojure.test :refer :all]
            [org.apache.clojure-mxnet.eval-metric :as eval-metric]
            [org.apache.clojure-mxnet.io :as mx-io]
            [org.apache.clojure-mxnet.module :as m]
            [org.apache.clojure-mxnet.optimizer :as optimizer]
            [org.apache.clojure-mxnet.symbol :as sym]
            [org.apache.clojure-mxnet.context :as context]
            [clojure.string :as string]))

;;; load the bert base model
(def model-dir "/Users/cmeier/workspace/deep-learning/bert-gluon/sentence_embedding")

(def bert-base (m/load-checkpoint {:prefix (str model-dir "/bert-base") :epoch 0}))
(def bert-base-sym (m/symbol bert-base))
(def arg-params (m/arg-params bert-base))
(def aux-params (m/aux-params bert-base))
(def num-classes 2)
(def fine-tune-mod (as-> bert-base-sym data
                     (sym/fully-connected "fc1" {:data data :num-hidden num-classes})
                     (sym/softmax-output "softmax" {:data data})))

(comment 
  (let [mod (-> (m/module msymbol {:contexts [(context/cpu)]})
                (m/bind {:data-shapes (mx-io/provide-data-desc train-iter) :label-shapes (mx-io/provide-label-desc val-iter)})
                (m/init-params {:arg-params arg-params :aux-params aux-params
                                :allow-missing true}))]
    (m/fit mod
           {:train-data train-iter
            :eval-data val-iter
            :num-epoch 1
            :fit-params (m/fit-params {:intializer (init/xavier {:rand-type "gaussian"
                                                                 :factor-type "in"
                                                                 :magnitude 2})
                                       :batch-end-callback (callback/speedometer batch-size 10)})}))


  (def train-text (slurp (str model-dir "/" "dev.tsv")))
  ;;; we want cols 3, 4, and 0
  (def train-data (let [rows (string/split train-text #"\n")]
                    (map (fn [row] (let [parsed-row (string/split row #"\t")]
                                     [(nth parsed-row 3)
                                      (nth parsed-row 4)
                                      (nth parsed-row 0)]))
                         (rest rows))))

  ;; next thing we want to do is tokenize the sequence

  (take 2 train-data)
  
  (-> train-text
      (string/split #"\n")
      (first))

  )



