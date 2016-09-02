#!/usr/bin/python
#-------------------------------------------------------------
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
#-------------------------------------------------------------
from sklearn import datasets, neighbors
from systemml.mllearn import LogisticRegression, LinearRegression, SVM, NaiveBayes
from pyspark.sql import SQLContext
from pyspark.context import SparkContext
import unittest
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF, Tokenizer
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics

sc = SparkContext()
sqlCtx = SQLContext(sc)

# Currently not integrated with JUnit test
# ~/spark-1.6.1-scala-2.11/bin/spark-submit --master local[*] --driver-class-path SystemML.jar test.py
class TestMLLearn(unittest.TestCase):
    def testLogisticSK1(self):
        digits = datasets.load_digits()
        X_digits = digits.data
        y_digits = digits.target
        n_samples = len(X_digits)
        X_train = X_digits[:.9 * n_samples]
        y_train = y_digits[:.9 * n_samples]
        X_test = X_digits[.9 * n_samples:]
        y_test = y_digits[.9 * n_samples:]
        logistic = LogisticRegression(sqlCtx)
        score = logistic.fit(X_train, y_train).score(X_test, y_test)
        self.failUnless(score > 0.9)

    def testLogisticSK2(self):
        digits = datasets.load_digits()
        X_digits = digits.data
        y_digits = digits.target
        n_samples = len(X_digits)
        X_train = X_digits[:.9 * n_samples]
        y_train = y_digits[:.9 * n_samples]
        X_test = X_digits[.9 * n_samples:]
        y_test = y_digits[.9 * n_samples:]
        # Convert to DataFrame for i/o: current way to transfer data
        logistic = LogisticRegression(sqlCtx, transferUsingDF=True)
        score = logistic.fit(X_train, y_train).score(X_test, y_test)
        self.failUnless(score > 0.9)

    def testLogisticMLPipeline1(self):
        training = sqlCtx.createDataFrame([
            (0, "a b c d e spark", 1.0),
            (1, "b d", 2.0),
            (2, "spark f g h", 1.0),
            (3, "hadoop mapreduce", 2.0),
            (4, "b spark who", 1.0),
            (5, "g d a y", 2.0),
            (6, "spark fly", 1.0),
            (7, "was mapreduce", 2.0),
            (8, "e spark program", 1.0),
            (9, "a e c l", 2.0),
            (10, "spark compile", 1.0),
            (11, "hadoop software", 2.0)
            ], ["id", "text", "label"])
        tokenizer = Tokenizer(inputCol="text", outputCol="words")
        hashingTF = HashingTF(inputCol="words", outputCol="features", numFeatures=20)
        lr = LogisticRegression(sqlCtx)
        pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])
        model = pipeline.fit(training)
        test = sqlCtx.createDataFrame([
            (12, "spark i j k", 1.0),
            (13, "l m n", 2.0),
            (14, "mapreduce spark", 1.0),
            (15, "apache hadoop", 2.0)], ["id", "text", "label"])
        result = model.transform(test)
        predictionAndLabels = result.select("prediction", "label")
        evaluator = MulticlassClassificationEvaluator()
        score = evaluator.evaluate(predictionAndLabels)
        self.failUnless(score == 1.0)

    def testLinearRegressionSK1(self):
        diabetes = datasets.load_diabetes()
        diabetes_X = diabetes.data[:, np.newaxis, 2]
        diabetes_X_train = diabetes_X[:-20]
        diabetes_X_test = diabetes_X[-20:]
        diabetes_y_train = diabetes.target[:-20]
        diabetes_y_test = diabetes.target[-20:]
        regr = LinearRegression(sqlCtx)
        regr.fit(diabetes_X_train, diabetes_y_train)
        score = regr.score(diabetes_X_test, diabetes_y_test)
        self.failUnless(score > 0.4) # TODO: Improve r2-score (may be I am using it incorrectly)

    def testLinearRegressionSK2(self):
        diabetes = datasets.load_diabetes()
        diabetes_X = diabetes.data[:, np.newaxis, 2]
        diabetes_X_train = diabetes_X[:-20]
        diabetes_X_test = diabetes_X[-20:]
        diabetes_y_train = diabetes.target[:-20]
        diabetes_y_test = diabetes.target[-20:]
        regr = LinearRegression(sqlCtx, transferUsingDF=True)
        regr.fit(diabetes_X_train, diabetes_y_train)
        score = regr.score(diabetes_X_test, diabetes_y_test)
        self.failUnless(score > 0.4) # TODO: Improve r2-score (may be I am using it incorrectly)

    def testSVMSK1(self):
        digits = datasets.load_digits()
        X_digits = digits.data
        y_digits = digits.target
        n_samples = len(X_digits)
        X_train = X_digits[:.9 * n_samples]
        y_train = y_digits[:.9 * n_samples]
        X_test = X_digits[.9 * n_samples:]
        y_test = y_digits[.9 * n_samples:]
        svm = SVM(sqlCtx, is_multi_class=True)
        score = svm.fit(X_train, y_train).score(X_test, y_test)
        self.failUnless(score > 0.9)

    def testSVMSK2(self):
        digits = datasets.load_digits()
        X_digits = digits.data
        y_digits = digits.target
        n_samples = len(X_digits)
        X_train = X_digits[:.9 * n_samples]
        y_train = y_digits[:.9 * n_samples]
        X_test = X_digits[.9 * n_samples:]
        y_test = y_digits[.9 * n_samples:]
        svm = SVM(sqlCtx, is_multi_class=True, transferUsingDF=True)
        score = svm.fit(X_train, y_train).score(X_test, y_test)
        self.failUnless(score > 0.9)

    def testNaiveBayesSK1(self):
        digits = datasets.load_digits()
        X_digits = digits.data
        y_digits = digits.target
        n_samples = len(X_digits)
        X_train = X_digits[:.9 * n_samples]
        y_train = y_digits[:.9 * n_samples]
        X_test = X_digits[.9 * n_samples:]
        y_test = y_digits[.9 * n_samples:]
        nb = NaiveBayes(sqlCtx)
        score = nb.fit(X_train, y_train).score(X_test, y_test)
        self.failUnless(score > 0.85)

    def testNaiveBayesSK2(self):
        categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
        newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
        newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)
        vectorizer = TfidfVectorizer()
        # Both vectors and vectors_test are SciPy CSR matrix
        vectors = vectorizer.fit_transform(newsgroups_train.data)
        vectors_test = vectorizer.transform(newsgroups_test.data)
        nb = NaiveBayes(sqlCtx)
        nb.fit(vectors, newsgroups_train.target)
        pred = nb.predict(vectors_test)
        score = metrics.f1_score(newsgroups_test.target, pred, average='weighted')
        self.failUnless(score > 0.8)


if __name__ == '__main__':
    unittest.main()
