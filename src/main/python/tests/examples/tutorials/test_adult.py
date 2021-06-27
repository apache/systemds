# -------------------------------------------------------------
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
# -------------------------------------------------------------
import os
import unittest

import numpy as np
from systemds.context import SystemDSContext
from systemds.examples.tutorials.adult import DataManager
from systemds.operator import OperationNode
from systemds.operator.algorithm import kmeans, multiLogReg, multiLogRegPredict, l2svm, confusionMatrix, scale, scaleApply, split, winsorize
from systemds.script_building import DMLScript


class Test_DMLScript(unittest.TestCase):
    """
    Test class for adult dml script tutorial code.
    """

    sds: SystemDSContext = None
    d: DataManager = None
    neural_net_src_path: str = "../../tests/examples/tutorials/neural_net_source.dml"
    dataset_path_train: str = "../../test/resources/datasets/adult/train_data.csv"
    dataset_path_train_mtd: str = "../../test/resources/datasets/adult/train_data.csv.mtd"
    dataset_path_test: str = "../../test/resources/datasets/adult/test_data.csv"
    dataset_path_test_mtd: str = "../../test/resources/datasets/adult/test_data.csv.mtd"
    dataset_jspec: str = "../../test/resources/datasets/adult/jspec.json"

    @classmethod
    def setUpClass(cls):
        cls.sds = SystemDSContext()
        cls.d = DataManager()

    @classmethod
    def tearDownClass(cls):
        cls.sds.close()

    def test_train_data(self):
        x = self.d.get_train_data()
        self.assertEqual((32561, 14), x.shape)

    def test_train_labels(self):
        y = self.d.get_train_labels()
        self.assertEqual((32561,), y.shape)

    def test_test_data(self):
        x_l = self.d.get_test_data()
        self.assertEqual((16281, 14), x_l.shape)

    def test_test_labels(self):
        y_l = self.d.get_test_labels()
        self.assertEqual((16281,), y_l.shape)

    def test_preprocess(self):
        #assumes certain preprocessing
        train_data, train_labels, test_data, test_labels = self.d.get_preprocessed_dataset()
        self.assertEqual((30162,104), train_data.shape)
        self.assertEqual((30162, ), train_labels.shape)
        self.assertEqual((15060,104), test_data.shape)
        self.assertEqual((15060, ), test_labels.shape)

    def test_multi_log_reg(self):
        # Reduced because we want the tests to finish a bit faster.
        train_count = 15000
        test_count = 5000

        train_data, train_labels, test_data, test_labels = self.d.get_preprocessed_dataset()

        # Train data
        X = self.sds.from_numpy( train_data[:train_count])
        Y = self.sds.from_numpy( train_labels[:train_count])
        Y = Y + 1.0

        # Test data
        Xt = self.sds.from_numpy(test_data[:test_count])
        Yt = self.sds.from_numpy(test_labels[:test_count])
        Yt = Yt + 1.0

        betas = multiLogReg(X, Y)

        [_, y_pred, acc] = multiLogRegPredict(Xt, betas, Yt).compute()

        self.assertGreater(acc, 80)

        confusion_matrix_abs, _ = confusionMatrix(self.sds.from_numpy(y_pred), Yt).compute()

        self.assertTrue(
            np.allclose(
                confusion_matrix_abs,
                np.array([[3503, 503],
                          [268, 726]])
            )
        )

    def test_neural_net(self):
        # Reduced because we want the tests to finish a bit faster.
        train_count = 15000
        test_count = 5000

        train_data, train_labels, test_data, test_labels = self.d.get_preprocessed_dataset(interpolate=True, standardize=True, dimred=0.1)

        # Train data
        X = self.sds.from_numpy( train_data[:train_count])
        Y = self.sds.from_numpy( train_labels[:train_count])

        # Test data
        Xt = self.sds.from_numpy(test_data[:test_count])
        Yt = self.sds.from_numpy(test_labels[:test_count])

        FFN_package = self.sds.source(self.neural_net_src_path, "fnn", print_imported_methods=True)

        network = FFN_package.train(X, Y, 1, 16, 0.01, 1)

        self.assertTrue(type(network) is not None) # sourcing and training seems to works

        FFN_package.save_model(network, '"model/python_FFN/"').compute(verbose=True)

        # TODO This does not work yet, not sure what the problem is
        #probs = FFN_package.predict(Xt, network).compute(True)
        # FFN_package.eval(Yt, Yt).compute()



    def test_level1(self):
        # Reduced because we want the tests to finish a bit faster.
        train_count = 15000
        test_count = 5000
        ################################################################################################################
        '''
        This example shows how one can work with Numpy data within the system ds framework. 
        More precisely we will make use of the build in DataManager, Multinomial Logistic Regression function and the 
        confusion Matrix function.
        The dataset used in this tutorial is a preprocessed version
        of the "UCI Adult Data Set". If you are interested in data preprocessing there is a separate tutorial on this topic.
        If one wants to skip the explanation then the full script is available at the bottom of this page.
        
        Our goal will be to predict whether the income of a person exceeds $50K/yr based on census data. The Adult 
        Dataset contains attributes like: age, workclass, education, marital-status, occupation, race, [...] and the labels
        >50K or <=50K. We will train a Multinomial Logistic Regression model on the training dataset and subsequently we 
        will use the test dataset to asses how well our model can predict if the income is above or below $50K/yr based on the
        attributes. 
        '''
        ################################################################################################################
        ################################################################################################################
        '''
        Step 1: Load and prepare data
        First we get our training and testing data from the build in DataManager. Since the multiLogReg function the 
        labels (Y) to be > 0 we add 1 to all labels. This ensures that the smallest label is >= 1. 
        '''
        ################################################################################################################
        train_data, train_labels, test_data, test_labels = self.d.get_preprocessed_dataset(interpolate=True,
                                                                                           standardize=True, dimred=0.1)
        #####Alternate way to get DataManager
        '''
        from systemds.examples.tutorials.mnist import DataManager
        d = DataManager()
        X = d.get_preprocessed_dataset()
        Y = d.get_preprocessed_dataset()
        '''
        #### Code
        # Train data
        X = self.sds.from_numpy(train_data[:train_count])
        Y = self.sds.from_numpy(train_labels[:train_count])
        Y = Y + 1.0

        # Test data
        Xt = self.sds.from_numpy(test_data[:test_count])
        Yt = self.sds.from_numpy(test_labels[:test_count])
        Yt = Yt + 1.0

        ################################################################################################################
        '''
        Step 2: Training
        Now that we prepared the data we can use the multiLogReg function. First we will train the model on our 
        training data. Afterwards we can make predictions on the test data and asses the performance of the model.
        '''
        ################################################################################################################
        #####Alternate way
        '''
        from systemds.operator.algorithm import multiLogReg
        bias = multiLogReg(X_ds, Y_ds)
        '''
        #### Code
        betas = multiLogReg(X, Y)
        ################################################################################################################
        '''
        Note that nothing has been calculated yet. In SystemDS the calculation is executed once .compute() is called. 
        E.g. betas_res = betas.compute(). 
        
        
        We can now use the trained model to make predictions on the test data. 
        '''
        ################################################################################################################
        [_, y_pred, acc] = multiLogRegPredict(Xt, betas, Yt).compute()
        self.assertGreater(acc, 80) #Todo remove?
        # todo add text how high acc should be with this config
        ################################################################################################################
        '''
        The multiLogRegPredict function has three return values:
            -m, a matrix with the mean probaility of correctly classifying each label. We not use it further in this example.
            -y_pred, is the predictions made using the model
            -acc, is the accuracy achieved by the model.
        '''
        ################################################################################################################
        ################################################################################################################
        '''
        Step 3: Confusion Matrix
        A confusion matrix is a useful tool to analyze the performance of the model and to obtain a better understanding
        which classes the model has difficulties to separate. 
        The confusionMatrix function takes the predicted labels and the true labels. It then returns the confusion matrix
        for the predictions and the confusion matrix averages of each true class.
        
        If you followed the tutorial you should be able to verify the results with the provided assertTrue function call. 
        
        '''
        ################################################################################################################


        confusion_matrix_abs, _ = confusionMatrix(self.sds.from_numpy(y_pred), Yt).compute()
        # todo print confusion matrix? Explain cm?
        self.assertTrue(
            np.allclose(
                confusion_matrix_abs,
                np.array([[3583, 502],
                          [245, 670]])
            )
        )

    def test_level2(self):
        #####################################################################################
        """" 
        Level 2:
        This part of the tutorial shows a more in depth overview for the preprocessing capabilites that systemds has to offer.
        We will take a new and raw dataset using the csv format and read it with systemds. Then do the heavy lifting for the prerpocessing with systemds. We will also show how to 
        switch from systemds to numpy and then back to systemds as this might be relevant for certain usecases.
        
        Step 1:
        First of all, we need to download the dataset and create a mtd-file for specifying different properties that the dataset has.
        We downloaded the train and test dataset from: https://archive.ics.uci.edu/ml/datasets/adult 
        The downloaded dataset has been slightly modified for convenience. These modifications entail removing unnecessary newlines at the end of the files,
        adding column names at the top of the file and finally removing dots at the end of each entry inside the test_dataset file.
        
        After these modifications we have to define a mtd-file for each file we want to read. This mtd file has to be in the same directory as the dataset.
        In this particular example, the dataset is split into two files "train_data.csv" and "test_data.csv". We want to read both, which means that we will define a mtd-file for 
        each of them. Those files will be called "train_data.csv.mtd" and "test_data.csv.mtd".
        In these files we can define certain properties that the file has and also specify which values are supposed to get treated like missing values.
        
        The content of the train_data.csv.mtd file is:
        ### Content Start
        {
        "data_type": "frame",
        "format": "csv",
        "header": true,
        "naStrings": [ "?", "" ],
        "rows": 32561,
        "cols": 15
        }
        ### Content END
        
        The "format" of the file is csv, and "header" is set to true because we added the feature names as headers to the csv files.
        "data_type" is set to frame as the preprocessing functions that we use require this datatype. 
        The value of "naStrings" is a list of all the String values that should be treated as unknown values during the preprocessing.
        Also, "rows" in our example is set to 32561, as we have this many entries and "cols" is set to 15 as we have this many features in our datasets.
        
         After these requirements are completed, we have to define a SystemDSContext for reading our dataset. We can do this in the following way.
        """""
        ############################################################################################

        #### General way to define SystemDSContext
        """"
        from systemds.context import SystemDSContext
        with SystemDSContext() as sds:
        """""
        #### END
        train_count = 30000
        test_count = 10000
        ####################################################################################
        """""
        With this context we can now define a read operation using the path of the dataset and a schema.
        The schema simply defines the datatypes for each colums.
        
        As already mentioned, systemds supports lazy execution by default, which means that the read operation is only executed after calling the compute() function.
        """""

        ################################################################################################

        SCHEMA = '"DOUBLE,STRING,DOUBLE,STRING,DOUBLE,STRING,STRING,STRING,STRING,STRING,DOUBLE,DOUBLE,DOUBLE,STRING,STRING"'

        F1 = self.sds.read(
            self.dataset_path_train,
            schema=SCHEMA
        )
        F2 = self.sds.read(
            self.dataset_path_test,
            schema=SCHEMA
        )

        ###################################################################################
        """""
        Step 2:
        Now that the datsets have been read we 
        """""

        ####################################################################################
        jspec = self.sds.read(self.dataset_jspec, data_type="scalar", value_type="string")
        # scaling does not have effect yet. We need to replace labels in test set with the same string as in train dataset
        X1, M1 = F1.rbind(F2).transform_encode(spec=jspec).compute()
        col_length = len(X1[0])
        X = X1[0:train_count, 0:col_length - 1]
        Y = X1[0:train_count, col_length - 1:col_length].flatten()
        # Test data
        Xt = X1[train_count:train_count + test_count, 0:col_length - 1]
        Yt = X1[train_count:train_count + test_count, col_length - 1:col_length].flatten()

        _, mean, sigma = scale(self.sds.from_numpy(X), True, True).compute()

        mean_copy = np.array(mean)
        sigma_copy = np.array(sigma)

        numerical_cols = []
        for count, col in enumerate(np.transpose(X)):
            for entry in col:
                if entry > 1 or entry < 0 or entry > 0 and entry < 1:
                    numerical_cols.append(count)
                    break

        for x in range(0, 105):
            if not x in numerical_cols:
                mean_copy[0][x] = 0
                sigma_copy[0][x] = 1

        mean = self.sds.from_numpy(mean_copy)
        sigma = self.sds.from_numpy(sigma_copy)
        X = self.sds.from_numpy(X)
        Y = self.sds.from_numpy(Y)
        Xt = self.sds.from_numpy(Xt)
        Yt = self.sds.from_numpy(Yt)
        X = scaleApply(winsorize(X, True), mean, sigma)
        Xt = scaleApply(winsorize(Xt, True), mean, sigma)
        # node = PROCESSING_split_package.m_split(X1,X1)
        # X,Y = node.compute()

        # Train data
        betas = multiLogReg(X, Y)

        [_, y_pred, acc] = multiLogRegPredict(Xt, betas, Yt).compute()

        self.assertGreater(acc, 80)
        confusion_matrix_abs, _ = confusionMatrix(self.sds.from_numpy(y_pred), Yt).compute()
        self.assertTrue(
            np.allclose(
                confusion_matrix_abs,
                np.array([[7073., 1011.],
                          [ 542., 1374.]])
            )
        )

    def test_level3(self):
        # Reduced because we want the tests to finish a bit faster.
        train_count = 30000
        test_count = 10000
        # self.sds.read(self.dataset_path_train, schema=self.dataset_path_train_mtd).compute(verbose=True)
        print("")

        SCHEMA = '"DOUBLE,STRING,DOUBLE,STRING,DOUBLE,STRING,STRING,STRING,STRING,STRING,DOUBLE,DOUBLE,DOUBLE,STRING,STRING"'

        F1 = self.sds.read(
            self.dataset_path_train,
            schema=SCHEMA
        )
        F2 = self.sds.read(
            self.dataset_path_test,
            schema=SCHEMA
        )
        jspec = self.sds.read(self.dataset_jspec, data_type="scalar", value_type="string")
        # scaling does not have effect yet. We need to replace labels in test set with the same string as in train dataset
        X1, M1 = F1.rbind(F2).transform_encode(spec=jspec).compute()
        col_length = len(X1[0])
        X = X1[0:train_count, 0:col_length - 1]
        Y = X1[0:train_count, col_length - 1:col_length].flatten() - 1
        # Test data
        Xt = X1[train_count:train_count + test_count, 0:col_length - 1]
        Yt = X1[train_count:train_count + test_count, col_length - 1:col_length].flatten() - 1

        _, mean, sigma = scale(self.sds.from_numpy(X), True, True).compute()

        mean_copy = np.array(mean)
        sigma_copy = np.array(sigma)

        numerical_cols = []
        for count, col in enumerate(np.transpose(X)):
            for entry in col:
                if entry > 1 or entry < 0 or entry > 0 and entry < 1:
                    numerical_cols.append(count)
                    break

        for x in range(0, 105):
            if not x in numerical_cols:
                mean_copy[0][x] = 0
                sigma_copy[0][x] = 1

        mean = self.sds.from_numpy(mean_copy)
        sigma = self.sds.from_numpy(sigma_copy)
        X = self.sds.from_numpy(X)
        Xt = self.sds.from_numpy(Xt)
        X = scaleApply(winsorize(X, True), mean, sigma)
        Xt = scaleApply(winsorize(Xt, True), mean, sigma)
        # node = PROCESSING_split_package.m_split(X1,X1)
        # X,Y = node.compute()

        # Train data

        FFN_package = self.sds.source(self.neural_net_src_path, "fnn", print_imported_methods=True)

        network = FFN_package.train(X, self.sds.from_numpy(Y), 1, 16, 0.01, 1)

        self.assertTrue(type(network) is not None)  # sourcing and training seems to works

        FFN_package.save_model(network, '"model/python_FFN/"').compute(verbose=True)

        # TODO This does not work yet, not sure what the problem is
        # FFN_package.eval(Yt, Yt).compute()"""





if __name__ == "__main__":
    unittest.main(exit=False)
