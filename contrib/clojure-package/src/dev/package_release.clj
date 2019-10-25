(ns dev.package-release
  (:require [clojure.java.io :as io]
            [clojure.edn :as edn]
            [clojure.java.shell :refer [sh]]
            [clojure.spec.alpha :as s]
            [clojure.string :as string]
            [dev.generator]))

(s/def ::build-type #{:osx-cpu :linux-gpu :linux-cpu})
(s/def ::release-number string?)
(s/def ::apache-gpg-key string?)
(s/def ::release-args (s/keys :req-un [::build-type ::release-number
                                       ::apache-gpg-key]))

(def scala-jar-name
  {:linux-cpu "org.apache.mxnet/mxnet-full_2.11-linux-x86_64-cpu"
   :linux-gpu "org.apache.mxnet/mxnet-full_2.11-linux-x86_64-gpu"
   :osx-cpu "org.apache.mxnet/mxnet-full_2.11-osx-x86_64-cpu"})

(defn backup-project-file
  "Backs up a copy of the original project file"
  []
  (do
    (sh "cp" "project.clj" "project.clj.bak")
    (sh "cp" "examples/imclassification/project.clj" "examples/imclassification/project.clj.bak")))

(defn make-new-project-file
  "Creates a new project file for the relase by
  changing the build type and the version along
  with the release number and the signer's public key
  which can be an email"
  [{:keys [build-type release-number apache-gpg-key]}]

  (let [project-data (read-string (slurp "project.clj"))
        project-header (take 3 project-data)
        project-rest-map (->> project-data
                              (drop 3)
                              (apply hash-map))
        new-header `[~'defproject
                     ~(symbol (str "org.apache.mxnet.contrib.clojure/clojure-mxnet"
                                   "-" (name build-type)))
                     ~release-number]
        new-dependencies (->> (:dependencies project-rest-map)
                              (remove #(string/includes? % "org.apache.mxnet"))
                              (into [[(symbol (scala-jar-name build-type)) release-number]]))
        new-project-rest-map     (-> project-rest-map
                                     (assoc :dependencies new-dependencies)
                                     (assoc-in [:deploy-repositories 0 1 :signing :gpg-key]
                                               apache-gpg-key))]

    (as-> (into [] new-project-rest-map) p
      (into new-header (mapcat identity p))
      (apply list p)
      (with-out-str (clojure.pprint/pprint p))
      (str dev.generator/license p)
      (spit "project.clj" p))))

(defn run-commands
  "Run shell commands and report exit and out"
  ([commands text]
   (commands text nil))
  ([commands text dir]
   (do
     (println "=====================  " text  "  =====================")
     (println "Please wait ....")
     (flush))
   (let [{:keys [out exit err]} (apply sh (if dir
                                            (into commands [:dir dir])
                                            commands))]
     (do
       (println out)
       (flush)
       (when-not (zero? exit)
         (println "Errors:")
         (println err)
         (flush))
       (zero? exit)))))

(defn reset-project-files
  "Resets the projects files back to the original saved version"
  []
  (do
    (sh "cp" "project.clj" "new-project-clj")
    (sh "cp" "project.clj.bak" "project.clj")
    (sh "cp" "examples/imclassification/project.clj.bak" "examples/imclassification/project.clj")))

(defn run-tests-and-install
  "With the project file in place, use lein to
  dowload deps, run tests and then install the generated
  jar"
  []
  (if (and (run-commands ["lein" "test"] "Running Tests")
           (run-commands ["lein" "install"] "Running Install"))
    (do
      (println "************************")
      (println "*****    SUCCESS   *****")
      (println "************************")
      (flush))
    (do
      (println "************************")
      (println "*****    FAILURE   *****")
      (println "************************")
      (flush))))


(defn clean-m2-release
  "This will move maven downloads of the release to test the
  download of the deployed jar from staging"
  [build-type release-number]
  (let [m2-path (str (System/getProperty "user.home")
                     "/.m2/repository/org/apache/mxnet/contrib/clojure/clojure-mxnet"
                     "-" build-type "/"
                     release-number)
        files-to-be-cleared (:out (sh "ls" m2-path))]
    (when files-to-be-cleared
      (println "Installed m2 jars found")
      (print files-to-be-cleared)
      (println "Do you want to mv the the m2 paths of " m2-path "to" (str m2-path "-bak") "?")
      (println "Type \"yes\" to confirm: ")
      (flush)
      (let [answer (read-line)]
        (if (= "yes" answer)
          (println (sh "mv" m2-path (str m2-path "-bak")))
          (println "Not moving"))))))

(defn test-installed-jar [build-type release-number]
  (let [project-data (read-string (slurp "./examples/imclassification/project.clj"))
        project-header (into [] (take 3 project-data))
        project-rest-map (->> project-data
                              (drop 3)
                              (apply hash-map))
        new-dependencies (->> (:dependencies project-rest-map)
                              (remove #(string/includes? % "org.apache.mxnet"))
                              (into [[(symbol (str "org.apache.mxnet.contrib.clojure/clojure-mxnet-" build-type)) release-number]]))
        new-project-rest-map     (-> project-rest-map
                                     (assoc :dependencies new-dependencies))]
    (do
      (as-> (into [] new-project-rest-map) p
        (into project-header (mapcat identity p))
        (apply list p)
        (with-out-str (clojure.pprint/pprint p))
        (str dev.generator/license p)
        (spit "./examples/imclassification/project.clj" p))
      (run-commands ["lein" "run"]
                    "Running Image Classification Example"
                    "examples/imclassification"))))


(defn run-build [args]
  (let [[build-type release-number apache-gpg-key] args
        release-args {:build-type (keyword build-type)
                      :release-number release-number
                      :apache-gpg-key apache-gpg-key}]
    
    (if (s/valid? ::release-args release-args)
      (try
        (do
          (test-installed-jar build-type release-number)
          (System/exit 0)
          )
        #_(do
            (make-new-project-file {:build-type :osx-cpu
                                    :release-number "1.5.1"
                                    :apache-gpg-key "cmeier@apache.org"} )
            (run-tests-and-install)
            (System/exit 0)
            )
        (finally (reset-project-files)))
      (do
        (println "Error with Args" release-args)
        (s/explain ::release-args release-args))
      )))

(defn -main [& args]
  (run-build args)
)


(comment



  (let [build-type "osx-cpu"
        release-number "1.5.1"
        project-data (read-string (slurp "./examples/imclassification/project.clj"))
        project-header (into [] (take 3 project-data))
        project-rest-map (->> project-data
                              (drop 3)
                              (apply hash-map))
        new-dependencies (->> (:dependencies project-rest-map)
                              (remove #(string/includes? % "org.apache.mxnet"))
                              (into [[(symbol (str "org.apache.mxnet.contrib.clojure/clojure-mxnet-" build-type)) release-number]]))
        new-project-rest-map     (-> project-rest-map
                                     (assoc :dependencies new-dependencies))]
    new-project-rest-map
    (do
      (as-> (into [] new-project-rest-map) p
        (into project-header (mapcat identity p))
        (apply list p)
        #_(with-out-str (clojure.pprint/pprint p))
        #_(str dev.generator/license p)
        #_(spit "./examples/imclassification/new-project.clj" p))
      #_(run-commands ["lein" "run"]
                    "Running Image Classification Example"
                    "examples/imclassification")))

  (run-build ["osx-cpu" "1.5.1" "cmeier@apache.org"])
  
  (let [m2-path (str (System/getProperty "user.home")
                 "/.m2/repository/org/apache/mxnet/contrib/clojure/clojure-mxnet")
        files-to-be-cleared (:out (sh "ls" m2-path))]
    files-to-be-cleared)

  (sh "cd" "examples")
  (sh "lein" "run" :dir "examples/imclassification")
  (sh "pwd")

  (sh "ls" "~")



"[[org.apache.mxnet.contrib.clojure/clojure-mxnet-osx-cpu \"1.5.1\"]]"
  
  )
