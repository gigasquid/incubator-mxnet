(ns org.apache.clojure-mxnet.kanerva
  (:require [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.ndarray-api :as ndarray-api]
            [org.apache.clojure-mxnet.ndarray-random-api :as random-api]
            [org.apache.clojure-mxnet.dtype :as dtype]
            [org.apache.clojure-mxnet.module :as m]
            [org.apache.clojure-mxnet.random :as random]
            [org.apache.clojure-mxnet.io :as mx-io]))

(def DIM 10000)


(defn rand-hdv []
  (random-api/uniform {:low -1 :high 1 :shape [DIM]}))

#_(defn rand-hdv []
  (random-api/normal {:loc 0 :scale 0.1 :shape [DIM]}))

;;; range is from 1 to -1 with larger being more similar
(defn cosine-sim [a b]
  (ndarray/div (ndarray-api/dot a b)
               (ndarray/* (ndarray-api/norm {:data a})
                          (ndarray-api/norm {:data b}))))

(defn sim
  "scalar form of cosine sim"
  [a b]
  (-> (cosine-sim a b)
      (ndarray/->vec)
      (first)))

(let [x (rand-hdv)
      y (rand-hdv)
      z (rand-hdv)]
  {:xx (sim x x)
   :xy (sim x y)
   :xz (sim x z)})
;=> {:xx 0.99999964, :xy 0.0098313, :xz -0.0011888654}


(defn add-pair [a b]
  (as-> (ndarray/+ a b) data
    (ndarray-api/clip {:data data :a-min -1 :a-max 1})))


(defn add [& xs]
  (reduce
   (fn [r x] (add-pair r x))
   xs))

(let [x (rand-hdv)
      y (rand-hdv)
      z (rand-hdv)
      xy (add x y)
      xyz (add x y z)]
  {:sim-x-xy (sim x xy)
   :sim-x-xyz (sim x xyz)
   :sim-xy-xyz (sim xy xyz)})

;=> {:sim-x-xy 0.707282, :sim-x-xyz 0.5707346, :sim-xy-xyz 0.81151086}

(defn subtract-pair [a b]
  (as-> (ndarray/- a b) data
    (ndarray-api/clip {:data data :a-min -1 :a-max 1})))

(defn subtract [& xs]
  (reduce
   (fn [r x] (subtract-pair r x))
   xs))

(let [x (rand-hdv)
      y (rand-hdv)
      z (rand-hdv)
      xy (add x y)
      xyz (add x y z)
      x1 (subtract xy y)
      z1 (subtract xyz xy)]
  {:sim-x1-x (sim x1 x)
   :sim-z1-z (sim z1 z)})
 ;=> {:sim-x1-x 0.99999946, :sim-z1-z 1.0000031}


;;; note same operation 

(defn mult [x a]
  (ndarray/* x a))

(defn bind [x a]
  (mult x a))

(defn unbind [x a]
  (mult x a))


;;; multipliction (binding) preserves distance but randomnizes direction
(let [x (rand-hdv)
      y (rand-hdv)
      xy (add x y)
      w (rand-hdv)
      zxy (bind w xy)
      zx (bind w x)]
  {:sim-xy-x (sim xy x)
   :sim-xy-y (sim xy y)
   :sim-xy-w (sim xy w)
   :sim-zxy-zx (sim zxy zx)
   :sim-zxy-x (sim zxy x)})

;; {:sim-xy-x 0.7080267,
;;  :sim-xy-y 0.71123475,
;;  :sim-xy-w 0.0037557934,
;;  :sim-zxy-zx 0.7134853,
;;  :sim-zxy-x -0.02917951}



;;; encoding a map
;; H = {x a,y b,z c}

;;; bind pair enoding then add them up
;;; get key from map "What is the value of x in h

(let [x (rand-hdv)
      y (rand-hdv)
      z (rand-hdv)
      a (rand-hdv)
      b (rand-hdv)
      c (rand-hdv)
      h  (add (bind x a)
              (bind y b)
              (bind z c))
      v (unbind h x)
      v1 (unbind h y)
      v2 (unbind h z)]
  (sort-by val {:sim-v-a (sim v a)
                :sim-v-b (sim v b)
                :sim-v-c (sim v c)
                :sim-v1-a (sim v1 a)
                :sim-v1-b (sim v1 b)
                :sim-v1-c (sim v1 c)
                :sim-v2-a (sim v2 a)
                :sim-v2-b (sim v2 b)
                :sim-v2-c (sim v2 c)}))

;; ([:sim-v1-a -0.0156876]
;;  [:sim-v-a -0.0030651577]
;;  [:sim-v2-b -0.0012383474]
;;  [:sim-v1-c 0.0023840778]
;;  [:sim-v-b 0.0028715762]
;;  [:sim-v-c 0.0037706625]
;;  [:sim-v2-a 0.008526781]
;;  [:sim-v1-b 0.0098520005]
;;  [:sim-v2-c 0.025100842])


(defn create-hdv-mem
  "Creates a HDV associated memory store dictionary for the given keys"
  [ks]
  (zipmap ks (repeatedly #(rand-hdv))))

(defn fetch-from-mem
  "Fetch the nearest neighbor vector from associative memory"
  [mem x]
  (->> (map (fn [[k v]] [k (sim v x)]) mem)
       (sort-by second)
       (last)))

(let [mem (create-hdv-mem [:x :y :z :a :b :c])
      h (add (bind (:x mem) (:a mem))
             (bind (:y mem) (:b mem))
             (bind (:z mem) (:c mem)))
      v (unbind h (:x mem))]
  (fetch-from-mem mem v))

 ;-> [:a 0.44644096]

(defn create-hdv-map
  "Creates a hdv map with the associated memory"
  [m]
  (reduce (fn [{:keys [mem value]}
               [k v]]
            (let [k-hdv (rand-hdv)
                  v-hdv (rand-hdv)]
              {:mem (merge mem {k k-hdv v v-hdv})
               :value (if value
                        (add value (bind k-hdv v-hdv))
                        v-hdv)}))
          {:mem {}
           :value (rand-hdv)}
          m))

(let [{:keys [mem value]} (create-hdv-map {:x :a :y :b :z :c})
      h value]
  (fetch-from-mem mem (unbind h (:x mem))))


(defn rotate [a]
  (ndarray-api/np-roll {:data a :shift 1}))

(defn counter-rotate [a]
  (ndarray-api/np-roll {:data a :shift -1}))

(defn sequence-pair [a b]
  (mult (rotate a) b))

(defn reverse-sequence-pair [a b]
  (mult (counter-rotate a) b))

(defn hdv-seq [xs]
  (reduce
   (fn [r x] (sequence-pair r x))
   xs))

(let [a (rand-hdv)
      b (rand-hdv)
      c (rand-hdv)
      ab (mult (rotate a) b)
      v (counter-rotate (mult b ab))]
  (sort-by val {:va (sim v a)
                :vb (sim v b)
                :vc (sim v c)}))

;;; find val A from ABC

(let [mem (create-hdv-mem [:a :b :c])
      abc (hdv-seq [(:a mem) (:b mem) (:c mem)])
      bc (hdv-seq [(:b mem) (:c mem)])
      v (counter-rotate (counter-rotate (mult abc bc)))]
  (fetch-from-mem mem v))

;;; find val C from ABC

(let [mem (create-hdv-mem [:a :b :c])
      abc (hdv-seq [(:a mem) (:b mem) (:c mem)])
      rra (rotate (rotate (:a mem)))
      rb (rotate (:b mem))
      v (mult abc (mult rra rb))]
  (fetch-from-mem mem v))




;;;; next try to encode mnist


(def data-dir "data/")
(def batch-size 1)



(def train-data (mx-io/mnist-iter {:image (str data-dir "train-images-idx3-ubyte")
                                   :label (str data-dir "train-labels-idx1-ubyte")
                                   :label-name "softmax_label"
                                   :data-shape [784]
                                   :label-shape [10]
                                   :batch-size batch-size
                                   :shuffle true
                                   :flat true
                                   :silent false
                                   :seed 10}))

(def test-data (mx-io/mnist-iter {:image (str data-dir "t10k-images-idx3-ubyte")
                                  :label (str data-dir "t10k-labels-idx1-ubyte")
                                  :data-shape [10]
                                  :batch-size batch-size
                                  :flat true
                                  :silent false}))

(def batch (mx-io/next train-data))


(ndarray/* (first (mx-io/batch-data batch)) 256)

(mx-io/batch-label batch)
;; need a dictionary of 256 colors to hdvs
;; add classes of 1-10 to another dictionary as well
;; then make a sequence of 784 hdv for each picture
;;; create label profile for each class
;;; add each sequence of 784 hdv to correct profile vector

;;; for test - encode pic (find what profile vec is closest)



