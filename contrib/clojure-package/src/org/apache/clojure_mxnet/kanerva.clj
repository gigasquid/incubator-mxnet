(ns org.apache.clojure-mxnet.kanerva
  (:require [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.ndarray-api :as ndarray-api]
            [org.apache.clojure-mxnet.ndarray-random-api :as random-api]
            [org.apache.clojure-mxnet.dtype :as dtype]
            [org.apache.clojure-mxnet.module :as m]))

(def DIM 10000)

;;; use with seed 1 -1 



(def x (random-api/uniform {:low -1 :high 1 :shape [5]}))
(ndarray/->vec x)
[0.50137234 -0.3691433 0.21566129 -0.27257848 -0.34990555]

(random-api/randint {:low 0 :high 2 :shape [3]})

(def y (ndarray/ceil x))
(ndarray/->vec y)
[1.0 -0.0 1.0 -0.0 -0.0]

(def z (ndarray-api/cast {:data y :dtype (str dtype/UINT8)}))

(ndarray/->vec z)

[1.0 0.0 1.0 0.0 0.0]


(ndarray-api/broadcast-logical-xor y y)


(ndarray-api/np-roll {:data (ndarray/array [1 2 3] [3]) :shift 2})

(ndarray-api/dot y y)


(defn random-hdv []
  (as-> (random-api/uniform {:low -1 :high 1 :shape [DIM]}) data
    (ndarray-api/ceil {:data data})))

(defn random-hdv []
  (as-> (random-api/normal {:loc 0.5 :scale 0.00001 :shape [DIM]}) data
        (ndarray-api/round {:data data})))

(defn cosine-dist [a b]
  (let [result (ndarray/div (ndarray-api/dot a b)
                            (ndarray/* (ndarray-api/norm {:data a})
                                       (ndarray-api/norm {:data b})))]
    (- 1 (first (ndarray/->vec result)))))

(def mod-2 (ndarray/array [2] [1]))

(defn hamming-dist [a b]
  (let [result (as-> (ndarray/broadcast-logical-xor a b) data
                 (ndarray-api/sum {:data data}))]
    (first (ndarray/->vec result))))



(def sent-m (ndarray/array [1 1 1 1 0 0 0 0 0] [9]))
(def sent-h (ndarray/array [0 0 1 1 1 1 0 0 0] [9]))
(def sent-w (ndarray/array [0 0 0 1 0 0 1 1 1] [9]))


(hamming-dist sent-m sent-h)

(def xx (random-hdv))
(def yy (random-hdv))
(def zz (random-hdv))

(cosine-sim yy xx)
(cosine-sim xx xx)
(hamming-dist yy xx)
(hamming-dist xx xx)

(defn xor-mult [a b]
  (ndarray/broadcast-logical-xor a b))

(defn mean-add [a b]
  (ndarray/round (ndarray/div (ndarray/+ a b) 2)))

(ndarray/round (mean-add sent-m sent-h))

(def u (ndarray/array [1 -2 4] [3]))
(def v (ndarray/array [3 0 2] [3]))
(def w (ndarray/array [1 3 5] [3]))

(ndarray/norm w)


(let [u (ndarray/array [1 0 0] [3])
      v (ndarray/array [1 0 1] [3])]
  (cosine-dist u v))
0.70710677

(cosine-dist sent-m sent-m)

(defn rotate [a n]
  (ndarray-api/np-roll {:data a :shift n}))

(rotate v 1)

(cosine-dist xx yy) ;=> 0.5002175867557526
(cosine-dist xx zz) ;=> 0.5118435323238373
(cosine-dist yy yy) ;=> 1.1920928955078125E-7


(def xxyy (mean-add xx yy))

(cosine-dist xxyy xx) ;=> 0.1846330761909485
(cosine-dist xxyy yy) ;=> 0.1807548999786377
(cosine-dist xxyy zz) ;=> 0.3921467065811157

(defn mean-subtract [a b]
  (ndarray/round (ndarray/div (ndarray/- a b) 2)))

(mean-subtract (mean-add sent-m sent-h)
          sent-h)

(cosine-dist zz (mean-subtract xxyy xx))

;;;; mult randomizes but preserves dist

(def x (random-hdv))
(def y (random-hdv))
(def a (random-hdv))
(def xa (xor-mult x a))
(def ya (xor-mult y a))

(cosine-dist xa ya) ;=>  0.4992964267730713
(cosine-dist x y);=>   0.49700456857681274

(hamming-dist xa ya);=>  4988.0
(hamming-dist x y) ;=> 4988.0


(def h (ndarray/array [1 -1 1 1] [4]))
(def w (ndarray/array [1 1 -1 -1] [4] ))
(ndarray/* w (ndarray/* h w))

;;multi distributes over addition
(cosine-dist (xor-mult a (mean-add x y))
             (mean-add (xor-mult x a)
                       (xor-mult x y)))

(cosine-dist x (xor-mult xa a))

(def s1 (xor-mult x (rotate y 1)))
(cosine-dist y (xor-mult s1 (rotate y -1)))


