(ns bert.util
  (:require [clojure.java.io :as io]
            [clojure.string :as string]
            [cheshire.core :as json]))

(defn break-out-punctuation [s str-match]
  (->> (string/split (str s "<punc>") (re-pattern (str "\\" str-match)))
       (map #(string/replace % "<punc>" str-match))))

(defn break-out-punctuations [s]
  (if-let [target-char (first (re-seq #"[.,?!]" s))]
    (break-out-punctuation s target-char)
    [s]))

(defn tokenize [s]
  (->> (string/split s #"\s+")
       (mapcat break-out-punctuations)
       (into [])))

(defn pad [tokens pad-item num]
  (if (>= (count tokens) num)
    tokens
    (into tokens (repeat (- num (count tokens)) pad-item))))

(defn get-vocab []
  (let [vocab (json/parse-stream (io/reader "model/vocab.json"))]
    {:idx->token (get vocab "idx_to_token")
     :token->idx (get vocab "token_to_idx")}))

(defn tokens->idxs [token->idx tokens]
  (let [unk-idx (get token->idx "[UNK]")]
   (mapv #(get token->idx % unk-idx) tokens)))

(defn idxs->tokens [idx->token idxs]
  (mapv #(get idx->token %) idxs))
