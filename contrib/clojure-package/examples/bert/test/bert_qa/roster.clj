(ns bert-qa.roster
  (:require  [clojure.test :as t]
             [clojure.string :as string]))

(require '[clojure.xml :as xml]
         '[clojure.zip :as zip])

;;convenience function, first seen at nakkaya.com later in clj.zip src
(defn zip-str [s]
  (zip/xml-zip 
      (xml/parse (java.io.ByteArrayInputStream. (.getBytes s)))))

(def x (zip-str (string/replace (slurp "ppmc-roster.txt") #"\<input type=\"checkbox\"\>"
                                "<input type=\"checkbox\"></input>")))


;;; format
(def people (->> (first x)
                 :content
      (map :content)
      (rest)
      (first)
      (map :content)
      (map (fn [x] [(-> (first x) :content first :content first :attrs :href (string/split #"\/") last)
                    (-> (nth x 2) :content first :content first)]))
      
      ))

(defn emit-people [persons]
  (doseq [[username fullname] persons]
    (xml/emit-element {:tag :tr :content [{:tag :td,:content ["."]}
                                          {:tag :td,:content [username]}
                                          {:tag :td,:content [fullname]}]})))

(emit-people people)
