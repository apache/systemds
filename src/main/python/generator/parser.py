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


from os import listdir

class FunctionParser(object):

    def __init__(self, path:str, ending:str='dml'):
        """
        @param path: path where to look for python scripts
        """
        super(FunctionParser, self).__init__()
        self.path = path
        self.ending = '.{ending}'.format(ending=ending)
        self.files()


    def parse_function(self, path:str):
        """
        @param path: path of file to parse
        parses function
        @return:
            {
                'function_name': 'some_name',
                'parameters': [('param1','type','default_value'), ...],
                'return_values': [('retval1', 'type'),...]
            }
        """
        raise NotImplementedError()

    def files():
        """
        generator function to find files in self.path, that end with self.ending
        """
        for f in listdir(self.path):
            if len(f) > 4:
                if f[-3:] == self.ending:
                    yield f

