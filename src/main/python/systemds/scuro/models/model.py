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

class Model:
    def __init__(self, name: str):
        """
        Parent class for models used to perform a given task
        :param name: Name of the model
        
        The classifier (clf) should be set in the fit method of each child class
        """
        self.name = name
        self.clf = None
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Fits a model to the training data
        """
        raise f'Fit method not implemented for {self.name}'
    
    def test(self, X_test, y_test):
        """
        Tests the classifier on a test or validation set
        """
        raise f'Test method not implemented for {self.name}'
    