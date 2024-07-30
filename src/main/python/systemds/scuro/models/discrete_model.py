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
from models.model import Model


class DiscreteModel(Model):
    def __init__(self):
        """
        Placeholder for a discrete model implementation
        """
        super().__init__('DiscreteModel')
        
    def fit(self, X_train, y_train):
        self.clf = None
        train_accuracy = 0
        return train_accuracy
    
    def test(self, X_test, y_test):
        test_accuracy = 0
        return test_accuracy
        
        