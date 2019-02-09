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
            [org.apache.clojure-mxnet.ndarray :as ndarray]
            [lightweight-conv.data-helper :as data-helper]))

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
  [{:keys [data num-in-channels num-out-channels kernel pad stride]}]
  (println "Carin num-in-channels" num-in-channels " num-out-channels " num-out-channels)
  ;; can this be dynamic?
  (let [channels (sym/split {:data data :axis 1 :num-outputs num-in-channels})
        _ (println "Carin channels " channels)
        depthwise-outs (into [] (for [i (range num-in-channels)]
                                  (sym/convolution {:data (sym/get channels i)
                                                    :kernel kernel
                                                    :stride stride
                                                    :pad pad
                                                    :num-filter 1})))
        _ (println "Carin deptwise outs" depthwise-outs)
        depthwise-out (sym/concat depthwise-outs)]
    (println "Carin depthwise out " depthwise-out)
    ;; pointwise convolution
    (sym/convolution {:data depthwise-out
                      :kernel [1 1]
                      :stride [1 1]
                      :pad [0 0]
                      :num-filter num-out-channels}))
  data
  )



;;https://aws.amazon.com/blogs/machine-learning/train-neural-machine-translation-models-with-sockeye/

;;https://github.com/awslabs/sockeye/blob/master/sockeye_contrib/autopilot/tasks.py
(defn get-symbol []
  (as-> (sym/variable "data") data

    (sym/convolution "conv1" {:data data :kernel [3 3] :num-filter 32 :stride [2 2]})
    (depthwise-separable-conv {:data data :num-in-channels 16 :num-out-channels 32
                               :kernel [3 3] :pad [1 1] :stride [1 1]})
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

#_(defn get-symbol []
  (as-> (sym/variable "data") data
    (sym/fully-connected "fc1" {:data data :num-hidden 128})
    #(glu-activation data)
    (sym/activation "relu1" {:data data :act-type "relu"})
    (sym/fully-connected "fc2" {:data data :num-hidden 64})
    (sym/activation "relu2" {:data data :act-type "relu"})
    (sym/fully-connected "fc3" {:data data :num-hidden 10})
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

(defn shuffle-data [test-num {:keys [data label sentence-count sentence-size vocab-size embedding-size pretrained-embedding]}]
  (println "Shuffling the data and splitting into training and test sets")
  (println {:sentence-count sentence-count
            :sentence-size sentence-size
            :vocab-size vocab-size
            :embedding-size embedding-size
            :pretrained-embedding pretrained-embedding})
  (let [shuffled (shuffle (map #(vector %1 %2) data label))
        train-num (- (count shuffled) test-num)
        training (into [] (take train-num shuffled))
        test (into [] (drop train-num shuffled))
        ;; has to be channel x y
        train-data-shape (if pretrained-embedding
                           [train-num 1 sentence-size embedding-size]
                           [train-num 1 sentence-size])
        ;; has to be channel x y
        test-data-shape (if pretrained-embedding
                           [test-num 1 sentence-size embedding-size]
                           [test-num 1 sentence-size])]
    {:training {:data  (ndarray/array (into [] (flatten (mapv first training)))
                                      train-data-shape)
                :label (ndarray/array (into [] (flatten (mapv last  training)))
                                      [train-num])}
     :test {:data  (ndarray/array (into [] (flatten (mapv first test)))
                                  test-data-shape)
            :label (ndarray/array (into [] (flatten (mapv last  test)))
                                  [test-num])}}))


(comment
  (def mr-dataset-path "data/mr-data") ;; the MR polarity dataset path
  (def max-examples 100)
  (def pretrained-embedding :glove)
  (def embedding-size 50)
  (def test-size 10)

  (def ms-dataset (data-helper/load-ms-with-embeddings mr-dataset-path max-examples embedding-size {:pretrained-embedding pretrained-embedding}))
  (def sentence-size (:sentence-size ms-dataset))
  (def vocab-size (:vocab-size ms-dataset))
  (def shuffled (shuffle-data test-size ms-dataset))
  (def word-train-data  (mx-io/ndarray-iter [(get-in shuffled [:training :data])]
                                            {:label [(get-in shuffled [:training :label])]
                                             :label-name "softmax_label"
                                             :data-batch-size batch-size
                                             :last-batch-handle "pad"}))
  (def word-test-data (mx-io/ndarray-iter [(get-in shuffled [:test :data])]
                                      {:label [(get-in  shuffled [:test :label])]
                                       :label-name "softmax_label"
                                       :data-batch-size batch-size
                                       :last-batch-handle "pad"}))

 (mx-io/provide-data word-train-data)

  (train 1)

  )

