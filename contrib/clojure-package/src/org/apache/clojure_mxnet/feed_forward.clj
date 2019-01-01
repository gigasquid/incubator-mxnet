(ns org.apache.clojure-mxnet.feed-forward
  (:require [clojure.java.io :as io]
            [clojure.java.shell :refer [sh]]
            [clojure.test :refer :all]
            [org.apache.clojure-mxnet.eval-metric :as eval-metric]
            [org.apache.clojure-mxnet.io :as mx-io]
            [org.apache.clojure-mxnet.module :as m]
            [org.apache.clojure-mxnet.optimizer :as optimizer]
            [org.apache.clojure-mxnet.symbol :as sym]
            [clojure.reflect :as r]
            [org.apache.clojure-mxnet.context :as context]
            [org.apache.clojure-mxnet.util :as util]
            [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.initializer :as initializer])
  (:import (org.apache.mxnet FeedForward Model)))

(def data-dir "data/")
(def batch-size 100)
(def num-epoch 1)


(def train-data (mx-io/mnist-iter {:image (str data-dir "train-images-idx3-ubyte")
                                   :label (str data-dir "train-labels-idx1-ubyte")
                                   :label-name "softmax_label"
                                   :data-shape [1 28 28]
                                   :label-shape [1 1 10]
                                   :batch-size batch-size
                                   :shuffle true
                                   :flat false
                                   :silent false
                                   :seed 10}))

(def test-data (mx-io/mnist-iter {:image (str data-dir "t10k-images-idx3-ubyte")
                                  :label (str data-dir "t10k-labels-idx1-ubyte")
                                  :data-shape [1 28 28]
                                  :batch-size batch-size
                                  :flat false
                                  :silent false}))
(defn get-symbol []
  (as-> (sym/variable "data") data

    (sym/convolution "conv1" {:data data :kernel [3 3] :num-filter 32 :stride [2 2]})
    (sym/batch-norm "bn1" {:data data})
    (sym/activation "relu1" {:data data :act-type "relu"})
    (sym/pooling "mp1" {:data data :kernel [2 2] :pool-type "max" :stride [2 2]}) (sym/convolution "conv2" {:data data :kernel [3 3] :num-filter 32 :stride [2 2]})
    (sym/batch-norm "bn2" {:data data})
    (sym/activation "relu2" {:data data :act-type "relu"})
    (sym/pooling "mp2" {:data data :kernel [2 2] :pool-type "max" :stride [2 2]})

    (sym/flatten "fl" {:data data})
    (sym/fully-connected "fc2" {:data data :num-hidden 10})
    (sym/softmax-output "softmax" {:data data})))

(defn- setup-builder [builder {:keys [context num-epoch epoch-size optimizer
                                      initializer batch-size arg-params allow-extra-params
                                      begin-epoch train-data eval-data eval-metric kv-store
                                      epoch-end-callback batch-end-callback] :as opts}]
      (do
        (when context (into-array [(.setContext builder context)]))
        (when num-epoch (.setNumEpoch builder num-epoch))
        (when epoch-size (.setEpochSize builder epoch-size))
        (when optimizer (.setOptimizer builder optimizer))
        (when initializer (.setIntializer builder initializer))
        (when batch-size (.setBatchSize builder batch-size))
        (when arg-params (.setArgParams builder (util/convert-symbol-map arg-params)))
        (when allow-extra-params (.setAllowExtraParams builder allow-extra-params))
        (when begin-epoch (.setBeginEpoch builder (int begin-epoch)))
        (when train-data (.setTrainData builder train-data))
        (when eval-data (.setEvalData builder eval-data))
        (when eval-metric (.setEvalMetric builder eval-metric))
        (when kv-store (.setKVStore builder kv-store))
        (when epoch-end-callback (.setEpochEndCallback builder epoch-end-callback))
        (when batch-end-callback (.setBatchEndCallback builder batch-end-callback))))


(defn setup [sym {:keys [contexts num-epoch epoch-size optimizer initializer
                         batch-size arg-params allow-extra-params begin-epoch train-epoch
                         eval-data eval-metric kv-store epoch-end-callback
                         batch-end-callback]
                  :or {contexts [(context/default-context)]} :as opts}]
  (let [builder (FeedForward/newBuilder sym)]
    (do
      (setup-builder builder opts)
      (.setup builder))))

(defn build [sym {:keys [context num-epoch epoch-size optimizer initializer
                         batch-size arg-params allow-extra-params begin-epoch
                         train-data eval-data eval-metric kv-store epoch-end-callback
                         batch-end-callback] :as opts}]
  (let [builder (FeedForward/newBuilder sym)]
    (do
      (setup-builder builder opts)
      (.build builder))))

(defn predict
  ([model data-iter]
   (predict model data-iter -1))
  ([model data-iter num-batch]
   (do (vec (.predict model data-iter num-batch)))))

;;; fit

(defn fit
  ([model train-data]
   (fit model train-data nil))
  ([model train-data eval-data]
   (doto model
     (.fit train-data eval-data))))

;;; save

(defn save
  ([model prefix]
   (do (.save model prefix (.save$default$2 model))))
  ([model prefix num-epoch]
   (do (.save model prefix num-epoch))))

;;; load

(defn load
  ([prefix epoch]
   (load prefix epoch {}))
  ([prefix epoch {:keys [contexts num-epoch epoch-size optimizer
                         initializer batch-size allow-extra-params]
                  :or {contexts [(context/default-context)]
                       num-epoch -1
                       epoch-size -1
                       optimizer (optimizer/sgd)
                       initializer (initializer/uniform 0.01)
                       batch-size 128
                       allow-extra-params false} :as opts}]
   (FeedForward/load prefix
                     (int epoch)
                     (into-array contexts)
                     (int num-epoch)
                     (int epoch-size)
                     optimizer
                     initializer
                     (int batch-size)
                     allow-extra-params)))

(comment

  (def x (build (get-symbol) {:num-epoch 1 :train-data train-data :eval-data test-data}))

  (def y (setup (get-symbol) {:num-epoch 1 :train-data train-data :eval-data test-data}))

  (fit y train-data)
  (count (ndarray/->vec (first (predict y test-data 1))))

  (def z (fit (load "carin" 1 {:num-epoch 3}) train-data))

  (save z "bob")
  (def w (fit (load "bob" 3 {:num-epoch 5}) train-data))


  (.fit y train-data nil)
  (save x "carin")

  (count (ndarray/->vec (first (.predict x test-data -1))))


  (save x "carin")
  (.save )
  (r/reflect FeedForward))

=
(r/reflect x)

