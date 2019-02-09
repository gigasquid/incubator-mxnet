(ns lightweight-conv.core
  (:require [clojure.java.io :as io]
            [clojure.java.shell :refer [sh]]
            [org.apache.clojure-mxnet.context :as context]
            [org.apache.clojure-mxnet.eval-metric :as eval-metric]
            [org.apache.clojure-mxnet.io :as mx-io]
            [org.apache.clojure-mxnet.module :as m]
            [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.optimizer :as optimizer]
            [org.apache.clojure-mxnet.symbol :as sym]
            [org.apache.clojure-mxnet.util :as util]
            [org.apache.clojure-mxnet.ndarray :as ndarray]))

(def data-dir "data/")
(def batch-size 10)

(when-not (.exists (io/file (str data-dir "train-images-idx3-ubyte")))
  (sh "../../scripts/get_mnist_data.sh"))
;; for save checkpoints load checkpoints
(io/make-parents "model/dummy.txt")

;;; Load the MNIST datasets
(defonce train-data (mx-io/mnist-iter {:image (str data-dir "train-images-idx3-ubyte")
                                       :label (str data-dir "train-labels-idx1-ubyte")
                                       :label-name "softmax_label"
                                       :data-shape [1 28 28]
                                       :label-shape [1 1 10]
                                       :batch-size batch-size
                                       :shuffle true
                                       :flat false
                                       :silent false
                                       :seed 10}))

(defonce test-data (mx-io/mnist-iter {:image (str data-dir "t10k-images-idx3-ubyte")
                                      :label (str data-dir "t10k-labels-idx1-ubyte")
                                      :data-shape [1 28 28]
                                      :batch-size batch-size
                                      :flat false
                                      :silent false}))


;; Gated Linear Unit - Dauphin et al. 2017
;; implementation from https://github.com/awslabs/sockeye/blob/master/sockeye/convolution.py#L165
(defn glu-activation
  [data]
  (let [gates (sym/split {:data data :axis 1 :num-outputs 2})
        gate-a (sym/get gates 0)
        gate-b (sym/get gates 1)
        gate-b-act (sym/activation {:data gate-b :act-type "sigmoid"})]
    (sym/broadcast-mul [gate-a gate-b-act])))

;; Depthwise Separable Convolution
;; reference https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d
;; implementation https://github.com/bruinxiong/Xception.mxnet/blob/master/symbol_xception.py
;;https://fairseq.readthedocs.io/en/latest/getting_started.html#training-a-new-model
(defn depthwise-separable-conv
  [{:keys [data num-in-channels num-out-channels kernel pad stride lightweight?]}]
  ;; can this be dynamic?
  (let [channels (sym/split {:data data :axis 1 :num-outputs num-in-channels})
        depthwise-outs (into [] (doall (for [i (range num-in-channels)]
                                   (sym/convolution {:data (sym/get channels i)
                                                     :kernel kernel
                                                     :stride stride
                                                     :pad pad
                                                     :num-filter 1}))))
        depthwise-out (sym/concat depthwise-outs)]
    (println "Carin light-weight? " lightweight?)
    ;; pointwise convolution
    (sym/convolution {:data (if lightweight?
                              (sym/softmax {:data depthwise-out})
                              depthwise-out)
                      :kernel [1 1]
                      :stride [1 1]
                      :pad [0 0]
                      :num-filter num-out-channels})))
;;https://arxiv.org/pdf/1901.10430.pdf
(defn lightweight-conv [{:keys [data num-in-channels num-out-channels kernel pad stride] :as params}]
  (depthwise-separable-conv (merge params {:lightweight? true})))



;;https://aws.amazon.com/blogs/machine-learning/train-neural-machine-translation-models-with-sockeye/

;;https://github.com/awslabs/sockeye/blob/master/sockeye_contrib/autopilot/tasks.py
(defn get-symbol []
  (as-> (sym/variable "data") data

    #_(sym/convolution "conv1" {:data data :kernel [3 3] :num-filter 32 :stride [2 2]})
    (lightweight-conv {:data data :num-in-channels 1 :num-out-channels 32
                       :kernel [3 3] :pad [3 3] :stride [2 2]})
    (sym/batch-norm "bn1" {:data data})
    (sym/activation "relu1" {:data data :act-type "relu"})
    (sym/pooling "mp1" {:data data :kernel [2 2] :pool-type "max" :stride [2 2]})
    (sym/convolution "conv2" {:data data :kernel [3 3] :num-filter 32 :stride [2 2]})
    (sym/batch-norm "bn2" {:data data})
    (sym/batch-norm "bn3" {:data data})
    (sym/activation "relu2" {:data data :act-type "relu"})
    (sym/pooling "mp2" {:data data :kernel [2 2] :pool-type "max" :stride [2 2]})

    (sym/flatten "fl" {:data data})
    (sym/fully-connected "fc2" {:data data :num-hidden 10})
    (sym/softmax-output "softmax" {:data data})))


(defn train [num-epoch]
  (let [devs [(context/cpu 0)]
        mod (m/module (get-symbol) {:contexts devs})]
    ;;; note only one function for training
    (m/fit mod {:train-data train-data :eval-data test-data :num-epoch num-epoch})

    ;;high level predict (just a dummy call but it returns a vector of results
    (m/predict mod {:eval-data test-data})

    ;;;high level score (returs the eval values)
    (let [score (m/score mod {:eval-data test-data :eval-metric (eval-metric/accuracy)})]
      (println "High level predict score is " score))))


(comment

  (train 1)

  )

