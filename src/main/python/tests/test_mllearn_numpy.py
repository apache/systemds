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

# To run:
#   - Python 2: `PYSPARK_PYTHON=python2 spark-submit --master local[*] --driver-class-path SystemML.jar test_mllearn_numpy.py`
#   - Python 3: `PYSPARK_PYTHON=python3 spark-submit --master local[*] --driver-class-path SystemML.jar test_mllearn_numpy.py`

# Make the `systemml` package importable
import os
import sys
path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../")
sys.path.insert(0, path)

import unittest

import numpy as np
from pyspark.context import SparkContext
from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.sql import SparkSession
from sklearn import datasets, metrics, neighbors
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, r2_score
from systemml.mllearn import LinearRegression, LogisticRegression, NaiveBayes, SVM
from sklearn import linear_model

sc = SparkContext()
sparkSession = SparkSession.builder.getOrCreate()
import os

def writeColVector(X, fileName):
	fileName = os.path.join(os.getcwd(), fileName)
	X.tofile(fileName, sep='\n')
	metaDataFileContent = '{ "data_type": "matrix", "value_type": "double", "rows":' + str(len(X)) + ', "cols": 1, "nnz": -1, "format": "csv", "author": "systemml-tests", "created": "0000-00-00 00:00:00 PST" }'
	with open(fileName+'.mtd', 'w') as text_file:
		text_file.write(metaDataFileContent)

def deleteIfExists(fileName):
	try:
		os.remove(fileName)
	except OSError:
		pass

# Currently not integrated with JUnit test
# ~/spark-1.6.1-scala-2.11/bin/spark-submit --master local[*] --driver-class-path SystemML.jar test.py
class TestMLLearn(unittest.TestCase):
    def test_logistic(self):
        digits = datasets.load_digits()
        X_digits = digits.data
        y_digits = digits.target
        n_samples = len(X_digits)
        X_train = X_digits[:int(.9 * n_samples)]
        y_train = y_digits[:int(.9 * n_samples)]
        X_test = X_digits[int(.9 * n_samples):]
        y_test = y_digits[int(.9 * n_samples):]
        logistic = LogisticRegression(sparkSession)
        logistic.fit(X_train, y_train)
        mllearn_predicted = logistic.predict(X_test)
        sklearn_logistic = linear_model.LogisticRegression()
        sklearn_logistic.fit(X_train, y_train)
        self.failUnless(accuracy_score(sklearn_logistic.predict(X_test), mllearn_predicted) > 0.95) # We are comparable to a similar algorithm in scikit learn
    
    def test_logistic_mlpipeline(self):
        training = sparkSession.createDataFrame([
            ("a b c d e spark", 1.0),
            ("b d", 2.0),
            ("spark f g h", 1.0),
            ("hadoop mapreduce", 2.0),
            ("b spark who", 1.0),
            ("g d a y", 2.0),
            ("spark fly", 1.0),
            ("was mapreduce", 2.0),
            ("e spark program", 1.0),
            ("a e c l", 2.0),
            ("spark compile", 1.0),
            ("hadoop software", 2.0)
            ], ["text", "label"])
        tokenizer = Tokenizer(inputCol="text", outputCol="words")
        hashingTF = HashingTF(inputCol="words", outputCol="features", numFeatures=20)
        lr = LogisticRegression(sparkSession)
        pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])
        model = pipeline.fit(training)
        test = sparkSession.createDataFrame([
            ("spark i j k", 1.0),
            ("l m n", 2.0),
            ("mapreduce spark", 1.0),
            ("apache hadoop", 2.0)], ["text", "label"])
        result = model.transform(test)
        predictionAndLabels = result.select("prediction", "label")
        from pyspark.ml.evaluation import MulticlassClassificationEvaluator
        evaluator = MulticlassClassificationEvaluator()
        score = evaluator.evaluate(predictionAndLabels)
        self.failUnless(score == 1.0)

    def test_linear_regression(self):
        diabetes = datasets.load_diabetes()
        diabetes_X = diabetes.data[:, np.newaxis, 2]
        diabetes_X_train = diabetes_X[:-20]
        diabetes_X_test = diabetes_X[-20:]
        diabetes_y_train = diabetes.target[:-20]
        diabetes_y_test = diabetes.target[-20:]
        regr = LinearRegression(sparkSession, solver='direct-solve')
        regr.fit(diabetes_X_train, diabetes_y_train)
        mllearn_predicted = regr.predict(diabetes_X_test)
        sklearn_regr = linear_model.LinearRegression()
        sklearn_regr.fit(diabetes_X_train, diabetes_y_train)
        self.failUnless(r2_score(sklearn_regr.predict(diabetes_X_test), mllearn_predicted) > 0.95) # We are comparable to a similar algorithm in scikit learn

    def test_linear_regression_cg(self):
        diabetes = datasets.load_diabetes()
        diabetes_X = diabetes.data[:, np.newaxis, 2]
        diabetes_X_train = diabetes_X[:-20]
        diabetes_X_test = diabetes_X[-20:]
        diabetes_y_train = diabetes.target[:-20]
        diabetes_y_test = diabetes.target[-20:]
        regr = LinearRegression(sparkSession, solver='newton-cg')
        regr.fit(diabetes_X_train, diabetes_y_train)
        mllearn_predicted = regr.predict(diabetes_X_test)
        sklearn_regr = linear_model.LinearRegression()
        sklearn_regr.fit(diabetes_X_train, diabetes_y_train)
        self.failUnless(r2_score(sklearn_regr.predict(diabetes_X_test), mllearn_predicted) > 0.95) # We are comparable to a similar algorithm in scikit learn
                
    def test_svm(self):
        digits = datasets.load_digits()
        X_digits = digits.data
        y_digits = digits.target
        n_samples = len(X_digits)
        X_train = X_digits[:int(.9 * n_samples)]
        y_train = y_digits[:int(.9 * n_samples)]
        X_test = X_digits[int(.9 * n_samples):]
        y_test = y_digits[int(.9 * n_samples):]
        svm = SVM(sparkSession, is_multi_class=True, tol=0.0001)
        mllearn_predicted = svm.fit(X_train, y_train).predict(X_test)
        from sklearn import linear_model, svm
        clf = svm.LinearSVC()
        sklearn_predicted = clf.fit(X_train, y_train).predict(X_test)
        accuracy = accuracy_score(sklearn_predicted, mllearn_predicted)
        evaluation = 'test_svm accuracy_score(sklearn_predicted, mllearn_predicted) was {}'.format(accuracy)
        self.failUnless(accuracy > 0.95, evaluation)

    def test_naive_bayes(self):
        digits = datasets.load_digits()
        X_digits = digits.data
        y_digits = digits.target
        n_samples = len(X_digits)
        X_train = X_digits[:int(.9 * n_samples)]
        y_train = y_digits[:int(.9 * n_samples)]
        X_test = X_digits[int(.9 * n_samples):]
        y_test = y_digits[int(.9 * n_samples):]
        nb = NaiveBayes(sparkSession)
        mllearn_predicted = nb.fit(X_train, y_train).predict(X_test)
        from sklearn.naive_bayes import MultinomialNB
        clf = MultinomialNB()
        sklearn_predicted = clf.fit(X_train, y_train).predict(X_test)
        self.failUnless(accuracy_score(sklearn_predicted, mllearn_predicted) > 0.95 )
        
    def test_naive_bayes1(self):
        categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
        newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
        newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)
        vectorizer = TfidfVectorizer()
        # Both vectors and vectors_test are SciPy CSR matrix
        vectors = vectorizer.fit_transform(newsgroups_train.data)
        vectors_test = vectorizer.transform(newsgroups_test.data)
        nb = NaiveBayes(sparkSession)
        mllearn_predicted = nb.fit(vectors, newsgroups_train.target).predict(vectors_test)
        from sklearn.naive_bayes import MultinomialNB
        clf = MultinomialNB()
        sklearn_predicted = clf.fit(vectors, newsgroups_train.target).predict(vectors_test)
        self.failUnless(accuracy_score(sklearn_predicted, mllearn_predicted) > 0.95 )


if __name__ == '__main__':
    unittest.main()
