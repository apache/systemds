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

__all__ = ['LinearRegression', 'LogisticRegression', 'SVM', 'NaiveBayes']

import numpy as np
from pyspark.ml import Estimator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import DataFrame
import sklearn as sk
from sklearn.metrics import accuracy_score, r2_score
from py4j.protocol import Py4JError
import traceback
from sklearn.preprocessing import LabelEncoder
import threading
import time
import math

from ..converters import *
from ..classloader import *

def assemble(sqlCtx, pdf, inputCols, outputCol):
    tmpDF = sqlCtx.createDataFrame(pdf, list(pdf.columns))
    assembler = VectorAssembler(inputCols=list(inputCols), outputCol=outputCol)
    return assembler.transform(tmpDF)

class BaseSystemMLEstimator(Estimator):
    features_col = 'features'
    label_col = 'label'
    
    def set_features_col(self, colName):
        """
        Sets the default column name for features of PySpark DataFrame.

        Parameters
        ----------
        colName: column name for features (default: 'features')
        """
        self.features_col = colName

    def set_label_col(self, colName):
        """
        Sets the default column name for features of PySpark DataFrame.

        Parameters
        ----------
        colName: column name for features (default: 'label')
        """
        self.label_col = colName

    
    def _fit_df(self):
        try:
            self.model = self.estimator.fit(self.X._jdf)
        except Py4JError:
            traceback.print_exc()
    
    def fit_df(self, X):
        self.X = X
        self._fit_df()
        self.X = None
        return self
    
    def _fit_numpy(self):
        try:
            self.model = self.estimator.fit(convertToMatrixBlock(self.sc, self.X), convertToMatrixBlock(self.sc, self.y))
        except Py4JError:
            traceback.print_exc()
                    
    def fit_numpy(self, X, y):
        self.X = X
        self.y = y
        self._fit_numpy()
        self.X = None
        self.y = None
        return self
        
    # Returns a model after calling fit(df) on Estimator object on JVM
    def _fit(self, X):
        """
        Invokes the fit method on Estimator object on JVM if X is PySpark DataFrame

        Parameters
        ----------
        X: PySpark DataFrame that contain the columns features_col (default: 'features') and label_col (default: 'label')
        """
        if hasattr(X, '_jdf') and self.features_col in X.columns and self.label_col in X.columns:
            return self.fit_df(X)
        else:
            raise Exception('Incorrect usage: Expected dataframe as input with features/label as columns')

    def fit(self, X, y=None, params=None):
        """
        Invokes the fit method on Estimator object on JVM if X and y are on of the supported data types

        Parameters
        ----------
        X: NumPy ndarray, Pandas DataFrame, scipy sparse matrix
        y: NumPy ndarray, Pandas DataFrame, scipy sparse matrix
        """
        if y is None:
            return self._fit(X)
        elif y is not None and isinstance(X, SUPPORTED_TYPES) and isinstance(y, SUPPORTED_TYPES):
            y = self.encode(y)
            if self.transferUsingDF:
                pdfX = convertToPandasDF(X)
                pdfY = convertToPandasDF(y)
                if getNumCols(pdfY) != 1:
                    raise Exception('y should be a column vector')
                if pdfX.shape[0] != pdfY.shape[0]:
                    raise Exception('Number of rows of X and y should match')
                colNames = pdfX.columns
                pdfX[self.label_col] = pdfY[pdfY.columns[0]]
                df = assemble(self.sqlCtx, pdfX, colNames, self.features_col).select(self.features_col, self.label_col)
                self.fit_df(df)
            else:
                numColsy = getNumCols(y)
                if numColsy != 1:
                    raise Exception('Expected y to be a column vector')
                self.fit_numpy(X, y)
            if self.setOutputRawPredictionsToFalse:
                self.model.setOutputRawPredictions(False)
            return self
        else:
            raise Exception('Unsupported input type')

    def transform(self, X):
        return self.predict(X)
    
    # Returns either a DataFrame or MatrixBlock after calling transform(X:MatrixBlock, y:MatrixBlock) on Model object on JVM
    def predict(self, X):
        """
        Invokes the transform method on Estimator object on JVM if X and y are on of the supported data types

        Parameters
        ----------
        X: NumPy ndarray, Pandas DataFrame, scipy sparse matrix or PySpark DataFrame
        """
        if isinstance(X, SUPPORTED_TYPES):
            if self.transferUsingDF:
                pdfX = convertToPandasDF(X)
                df = assemble(self.sqlCtx, pdfX, pdfX.columns, self.features_col).select(self.features_col)
                retjDF = self.model.transform(df._jdf)
                retDF = DataFrame(retjDF, self.sqlCtx)
                retPDF = retDF.sort('__INDEX').select('prediction').toPandas()
                if isinstance(X, np.ndarray):
                    return self.decode(retPDF.as_matrix().flatten())
                else:
                    return self.decode(retPDF)
            else:
                try:
                    retNumPy = self.decode(convertToNumPyArr(self.sc, self.model.transform(convertToMatrixBlock(self.sc, X))))
                except Py4JError:
                    traceback.print_exc()
                if isinstance(X, np.ndarray):
                    return retNumPy
                else:
                    return retNumPy # TODO: Convert to Pandas
        elif hasattr(X, '_jdf'):
            if self.features_col in X.columns:
                # No need to assemble as input DF is likely coming via MLPipeline
                df = X
            else:
                assembler = VectorAssembler(inputCols=X.columns, outputCol=self.features_col)
                df = assembler.transform(X)
            retjDF = self.model.transform(df._jdf)
            retDF = DataFrame(retjDF, self.sqlCtx)
            # Return DF
            return retDF.sort('__INDEX')
        else:
            raise Exception('Unsupported input type')


class BaseSystemMLClassifier(BaseSystemMLEstimator):

    def encode(self, y):
        self.le = LabelEncoder()
        self.le.fit(y)
        return self.le.transform(y) + 1
        
    def decode(self, y):
        if self.le is not None:
            return self.le.inverse_transform(np.asarray(y - 1, dtype=int))
        else:
            return [ self.labelMap[int(i)] for i in y ]
        
    def score(self, X, y):
        """
        Scores the predicted value with ground truth 'y'

        Parameters
        ----------
        X: NumPy ndarray, Pandas DataFrame, scipy sparse matrix
        y: NumPy ndarray, Pandas DataFrame, scipy sparse matrix
        """
        return accuracy_score(y, self.predict(X))


class BaseSystemMLRegressor(BaseSystemMLEstimator):

    def encode(self, y):
        return y
        
    def decode(self, y):
        return y
    
    def score(self, X, y):
        """
        Scores the predicted value with ground truth 'y'

        Parameters
        ----------
        X: NumPy ndarray, Pandas DataFrame, scipy sparse matrix
        y: NumPy ndarray, Pandas DataFrame, scipy sparse matrix
        """
        return r2_score(y, self.predict(X), multioutput='variance_weighted')


class LogisticRegression(BaseSystemMLClassifier):
    """
    Performs both binomial and multinomial logistic regression.

    Examples
    --------
    
    Scikit-learn way
    
    >>> from sklearn import datasets, neighbors
    >>> from systemml.mllearn import LogisticRegression
    >>> from pyspark.sql import SQLContext
    >>> sqlCtx = SQLContext(sc)
    >>> digits = datasets.load_digits()
    >>> X_digits = digits.data
    >>> y_digits = digits.target + 1
    >>> n_samples = len(X_digits)
    >>> X_train = X_digits[:.9 * n_samples]
    >>> y_train = y_digits[:.9 * n_samples]
    >>> X_test = X_digits[.9 * n_samples:]
    >>> y_test = y_digits[.9 * n_samples:]
    >>> logistic = LogisticRegression(sqlCtx)
    >>> print('LogisticRegression score: %f' % logistic.fit(X_train, y_train).score(X_test, y_test))
    
    MLPipeline way
    
    >>> from pyspark.ml import Pipeline
    >>> from systemml.mllearn import LogisticRegression
    >>> from pyspark.ml.feature import HashingTF, Tokenizer
    >>> from pyspark.sql import SQLContext
    >>> sqlCtx = SQLContext(sc)
    >>> training = sqlCtx.createDataFrame([
    >>>     (0L, "a b c d e spark", 1.0),
    >>>     (1L, "b d", 2.0),
    >>>     (2L, "spark f g h", 1.0),
    >>>     (3L, "hadoop mapreduce", 2.0),
    >>>     (4L, "b spark who", 1.0),
    >>>     (5L, "g d a y", 2.0),
    >>>     (6L, "spark fly", 1.0),
    >>>     (7L, "was mapreduce", 2.0),
    >>>     (8L, "e spark program", 1.0),
    >>>     (9L, "a e c l", 2.0),
    >>>     (10L, "spark compile", 1.0),
    >>>     (11L, "hadoop software", 2.0)
    >>> ], ["id", "text", "label"])
    >>> tokenizer = Tokenizer(inputCol="text", outputCol="words")
    >>> hashingTF = HashingTF(inputCol="words", outputCol="features", numFeatures=20)
    >>> lr = LogisticRegression(sqlCtx)
    >>> pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])
    >>> model = pipeline.fit(training)
    >>> test = sqlCtx.createDataFrame([
    >>>     (12L, "spark i j k"),
    >>>     (13L, "l m n"),
    >>>     (14L, "mapreduce spark"),
    >>>     (15L, "apache hadoop")], ["id", "text"])
    >>> prediction = model.transform(test)
    >>> prediction.show()
    
    """
    
    def __init__(self, sqlCtx, penalty='l2', fit_intercept=True, max_iter=100, max_inner_iter=0, tol=0.000001, C=1.0, solver='newton-cg', transferUsingDF=False):
        """
        Performs both binomial and multinomial logistic regression.
        
        Parameters
        ----------
        sqlCtx: PySpark SQLContext
        penalty: Only 'l2' supported
        fit_intercept: Specifies whether to add intercept or not (default: True)
        max_iter: Maximum number of outer (Fisher scoring) iterations (default: 100)
        max_inner_iter: Maximum number of inner (conjugate gradient) iterations, or 0 if no maximum limit provided (default: 0)
        tol: Tolerance used in the convergence criterion (default: 0.000001)
        C: 1/regularization parameter (default: 1.0)
        solver: Only 'newton-cg' solver supported
        """
        self.sqlCtx = sqlCtx
        self.sc = sqlCtx._sc
        createJavaObject(self.sc, 'dummy')
        self.uid = "logReg"
        self.estimator = self.sc._jvm.org.apache.sysml.api.ml.LogisticRegression(self.uid, self.sc._jsc.sc())
        self.estimator.setMaxOuterIter(max_iter)
        self.estimator.setMaxInnerIter(max_inner_iter)
        if C <= 0:
            raise Exception('C has to be positive')
        reg = 1.0 / C
        self.estimator.setRegParam(reg)
        self.estimator.setTol(tol)
        self.estimator.setIcpt(int(fit_intercept))
        self.transferUsingDF = transferUsingDF
        self.setOutputRawPredictionsToFalse = True
        if penalty != 'l2':
            raise Exception('Only l2 penalty is supported')
        if solver != 'newton-cg':
            raise Exception('Only newton-cg solver supported')


class LinearRegression(BaseSystemMLRegressor):
    """
    Performs linear regression to model the relationship between one numerical response variable and one or more explanatory (feature) variables.
    
    Examples
    --------
    
    >>> import numpy as np
    >>> from sklearn import datasets
    >>> from systemml.mllearn import LinearRegression
    >>> from pyspark.sql import SQLContext
    >>> # Load the diabetes dataset
    >>> diabetes = datasets.load_diabetes()
    >>> # Use only one feature
    >>> diabetes_X = diabetes.data[:, np.newaxis, 2]
    >>> # Split the data into training/testing sets
    >>> diabetes_X_train = diabetes_X[:-20]
    >>> diabetes_X_test = diabetes_X[-20:]
    >>> # Split the targets into training/testing sets
    >>> diabetes_y_train = diabetes.target[:-20]
    >>> diabetes_y_test = diabetes.target[-20:]
    >>> # Create linear regression object
    >>> regr = LinearRegression(sqlCtx, solver='newton-cg')
    >>> # Train the model using the training sets
    >>> regr.fit(diabetes_X_train, diabetes_y_train)
    >>> # The mean square error
    >>> print("Residual sum of squares: %.2f" % np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2))
    
    """
    
    
    def __init__(self, sqlCtx, fit_intercept=True, max_iter=100, tol=0.000001, C=1.0, solver='newton-cg', transferUsingDF=False):
        """
        Performs linear regression to model the relationship between one numerical response variable and one or more explanatory (feature) variables.

        Parameters
        ----------
        sqlCtx: PySpark SQLContext
        fit_intercept: Specifies whether to add intercept or not (default: True)
        max_iter: Maximum number of conjugate gradient iterations, or 0 if no maximum limit provided (default: 100)
        tol: Tolerance used in the convergence criterion (default: 0.000001)
        C: 1/regularization parameter (default: 1.0)
        solver: Supports either 'newton-cg' or 'direct-solve' (default: 'newton-cg').
        Depending on the size and the sparsity of the feature matrix, one or the other solver may be more efficient.
        'direct-solve' solver is more efficient when the number of features is relatively small (m < 1000) and
        input matrix X is either tall or fairly dense; otherwise 'newton-cg' solver is more efficient.
        """
        self.sqlCtx = sqlCtx
        self.sc = sqlCtx._sc
        createJavaObject(self.sc, 'dummy')
        self.uid = "lr"
        if solver == 'newton-cg' or solver == 'direct-solve':
            self.estimator = self.sc._jvm.org.apache.sysml.api.ml.LinearRegression(self.uid, self.sc._jsc.sc(), solver)
        else:
            raise Exception('Only newton-cg solver supported')
        self.estimator.setMaxIter(max_iter)
        if C <= 0:
            raise Exception('C has to be positive')
        reg = 1.0 / C
        self.estimator.setRegParam(reg)
        self.estimator.setTol(tol)
        self.estimator.setIcpt(int(fit_intercept))
        self.transferUsingDF = transferUsingDF
        self.setOutputRawPredictionsToFalse = False


class SVM(BaseSystemMLClassifier):
    """
    Performs both binary-class and multiclass SVM (Support Vector Machines).

    Examples
    --------
    
    >>> from sklearn import datasets, neighbors
    >>> from systemml.mllearn import SVM
    >>> from pyspark.sql import SQLContext
    >>> sqlCtx = SQLContext(sc)
    >>> digits = datasets.load_digits()
    >>> X_digits = digits.data
    >>> y_digits = digits.target 
    >>> n_samples = len(X_digits)
    >>> X_train = X_digits[:.9 * n_samples]
    >>> y_train = y_digits[:.9 * n_samples]
    >>> X_test = X_digits[.9 * n_samples:]
    >>> y_test = y_digits[.9 * n_samples:]
    >>> svm = SVM(sqlCtx, is_multi_class=True)
    >>> print('LogisticRegression score: %f' % svm.fit(X_train, y_train).score(X_test, y_test))
     
    """


    def __init__(self, sqlCtx, fit_intercept=True, max_iter=100, tol=0.000001, C=1.0, is_multi_class=False, transferUsingDF=False):
        """
        Performs both binary-class and multiclass SVM (Support Vector Machines).

        Parameters
        ----------
        sqlCtx: PySpark SQLContext
        fit_intercept: Specifies whether to add intercept or not (default: True)
        max_iter: Maximum number iterations (default: 100)
        tol: Tolerance used in the convergence criterion (default: 0.000001)
        C: 1/regularization parameter (default: 1.0)
        is_multi_class: Specifies whether to use binary-class SVM or multi-class SVM algorithm (default: False)
        """
        self.sqlCtx = sqlCtx
        self.sc = sqlCtx._sc
        self.uid = "svm"
        createJavaObject(self.sc, 'dummy')
        self.estimator = self.sc._jvm.org.apache.sysml.api.ml.SVM(self.uid, self.sc._jsc.sc(), is_multi_class)
        self.estimator.setMaxIter(max_iter)
        if C <= 0:
            raise Exception('C has to be positive')
        reg = 1.0 / C
        self.estimator.setRegParam(reg)
        self.estimator.setTol(tol)
        self.estimator.setIcpt(int(fit_intercept))
        self.transferUsingDF = transferUsingDF
        self.setOutputRawPredictionsToFalse = False


class NaiveBayes(BaseSystemMLClassifier):
    """
    Performs Naive Bayes.

    Examples
    --------
    
    >>> from sklearn.datasets import fetch_20newsgroups
    >>> from sklearn.feature_extraction.text import TfidfVectorizer
    >>> from systemml.mllearn import NaiveBayes
    >>> from sklearn import metrics
    >>> from pyspark.sql import SQLContext
    >>> sqlCtx = SQLContext(sc)
    >>> categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
    >>> newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
    >>> newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)
    >>> vectorizer = TfidfVectorizer()
    >>> # Both vectors and vectors_test are SciPy CSR matrix
    >>> vectors = vectorizer.fit_transform(newsgroups_train.data)
    >>> vectors_test = vectorizer.transform(newsgroups_test.data)
    >>> nb = NaiveBayes(sqlCtx)
    >>> nb.fit(vectors, newsgroups_train.target)
    >>> pred = nb.predict(vectors_test)
    >>> metrics.f1_score(newsgroups_test.target, pred, average='weighted')

    """
    
    def __init__(self, sqlCtx, laplace=1.0, transferUsingDF=False):
        """
        Performs Naive Bayes.

        Parameters
        ----------
        sqlCtx: PySpark SQLContext
        laplace: Laplace smoothing specified by the user to avoid creation of 0 probabilities (default: 1.0)
        """
        self.sqlCtx = sqlCtx
        self.sc = sqlCtx._sc
        self.uid = "nb"
        createJavaObject(self.sc, 'dummy')
        self.estimator = self.sc._jvm.org.apache.sysml.api.ml.NaiveBayes(self.uid, self.sc._jsc.sc())
        self.estimator.setLaplace(laplace)
        self.transferUsingDF = transferUsingDF
        self.setOutputRawPredictionsToFalse = False