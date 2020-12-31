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


class PythonAPIFileGenerator(object):

    def __init__(self):
        super(PythonAPIFileGenerator, self).__init__()
        self.path = 'src/main/python/systemds/operator/algorithm'

    def generate_file(self, data:dict):
        """
        Generates file in self.path with name file_name
        and given file_contents as content
        @param data: dictionary containing
            {
                'file_name':'some_name',
                'file_contents': 'some_content'
            }
        """
        raise NotImplementedError()

class PythonAPIFunctionGenerator(object):

    def __init__(self):
        super(PythonAPIFunctionGenerator, self).__init__()

    def generate_function(self, data:dict) -> str:
        """
        Generates function definition for PythonAPI
        @param data:
            {
                'function_name': 'some_name',
                'function_header': 'header contained in \"\"\"'
                'parameters': [('param1','type','default_value'), ...],
                'return_values': [('retval1', 'type'),...]
            }
        @return: function definition
        """
        raise NotImplementedError()

class PythonAPIDocumentationGenerator(object):
    
    def __init__(self):
        super(PythonAPIDocumentationGenerator, self).__init__()

    def generate_documentation(self, data:dict) -> str:
        """
        Generates function header for PythonAPI
        @param data:
            {
                'function_name': 'some_name',
                'parameters': [('param1','description'), ...],
                'return_values': [('retval1', 'descritpion'),...]
            }
        @return: function header including '\"\"\"' at start and end
        """
        raise NotImplementedError()




