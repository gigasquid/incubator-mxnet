(ns org.apache.clojure-mxnet.infer.classifier-test
  (:require [org.apache.clojure-mxnet.infer :as infer]
            [org.apache.clojure-mxnet.dtype :as dtype]
            [org.apache.clojure-mxnet.io :as mx-io]
            [org.apache.clojure-mxnet.layout :as layout]
            [org.apache.clojure-mxnet.context :as context]
            [clojure.java.shell :refer [sh]]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [clojure.test :refer :all]))


(def data-dir "data/")
(def resnet-dir (str data-dir "resnet-18"))

(when-not (.exists (io/file resnet-dir))
  (sh "scripts/get_resnet_18_data.sh"))

(deftest test-single-classificaiton
 (let [descriptors [(mx-io/data-desc {:name "data"
                                      :shape [1 3 224 224]
                                      :layout layout/NCHW
                                      :dtype dtype/FLOAT32})]
       factory (infer/model-factory (str resnet-dir "/resnet-18") descriptors)
       classifier (infer/create-image-classifier factory)
       image (infer/load-image-from-file "test/test-images/kitten.jpg")
       [predictions] (infer/classify-image classifier image 5)]
   (is (some? predictions))
   (is (= 5 (count predictions)))
   (is (every? #(= 2 (count %)) predictions))
   (is (every? #(string? (first %)) predictions))
   (is (every? #(float? (second %)) predictions))
   (is (every? #(< 0 (second %) 1) predictions))
   (is (= ["n02123159 tiger cat"
           "n02124075 Egyptian cat"
           "n02123045 tabby, tabby cat"
           "n02127052 lynx, catamount"
           "n02128757 snow leopard, ounce, Panthera uncia"]
          (map first predictions)))))

