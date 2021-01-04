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
import re

class FunctionParser(object):

    header_input_pattern = r"^[ \t]*[#]+[ \t]*input[ \t\w:;.,#]*[\s#\-]*[#]+[\w\s\d:,.()\" \t\-]*[\s#\-]*$"
    header_output_pattern = r"[\s#\-]*[#]+[ \t]*(return|output)[ \t\w:;.,#]*[\s#\-]*[#]+[\w\s\d:,.()\" \t\-]*[\s#\-]*$"
    function_pattern = r"^m_[\w]+[ \t]+=[ \t]+function[^#{]*$"
    parameter_pattern = r"^m_[\w]+[\s]+=[\s]+function\(([\w\[\]\s,\d=.\-]*)\)[\s]*return[\s]*\(([\w\[\]\s,\d=.\-]*)\)"
    path_delimiter = "/"

    def __init__(self, path:str, ending:str='dml'):
        """
        @param path: path where to look for python scripts
        """
        super(FunctionParser, self).__init__()
        self.path = self.normalize_path(path)
        self.ending = '.{ending}'.format(ending=ending)
        self.files()

    def normalize_path(self, path:str):
        if len(path) > 0:
            path = "{path}{delim}".format(path=path, delim=self.__class__.path_delimiter)
        elif path[-1] != self.__class__.path_delimiter:
            path = "{path}{delim}".format(path=path, delim=self.__class__.path_delimiter)
        return path

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
        function_name = path.replace(self.ending, "")
        function_name = function_name.replace(self.path, "")
        function_definition = self.find_function_definition(path)
        print(function_definition)
        pattern = re.compile(self.__class__.parameter_pattern, flags=re.I|re.M)
        match = pattern.match(function_definition)
        param_str,retval_str = match.group(1,2)
        parameters = self.get_parameters(param_str)
        return_values = self.get_parameters(retval_str)
        data = {'function_name': function_name, 'parameters': parameters, 'return_values':return_values}
        return data
    
    def get_parameters(self, param_str:str):
        #pattern = re.compile(r"[\r\v\n\t]")
        #param_str = pattern.sub(" ", param_str)
        #print(param_str)
        params = re.split(r",[\s]*", param_str)
        parameters = []
        for param in params:
            splitted = re.split(r"[\s]+", param)
            dml_type = splitted[0]
            name = splitted[1]
            default_value = None
            if len(splitted) == 4:
                if splitted[2] == "=":
                    default_value = splitted[3]
            parameters.append((name, dml_type, default_value))
        return parameters

    def parse_header(self, path:str):
        """
        @param path: path of file to parse
        parses function
        @return:
            {
                'function_name': 'some_name',
                'parameters': [('param1','description'), ...],
                'return_values': [('retval1', 'description'),...]
            }
        """
        data = {'function_name': None, 'parameters': [], 'return_values':[]}
   

    def find_header_input_params(self, path:str):
        with open(path, 'r') as f:
            content = f.read()
        start = re.search(pattern=self.__class__.header_input_pattern, string=content, flags=re.I | re.M).end()
        end = re.search(pattern=self.__class__.header_output_pattern, string=content, flags=re.I | re.M).start()
        header = content[start:end]
        return header

    def find_function_definition(self, path:str):
        with open(path, 'r') as f:
            content = f.read()
        match = re.search(pattern=self.__class__.function_pattern, string=content, flags=re.I | re.M)
        start =match.start()
        end = match.end()
        return content[start:end]

    def files(self):
        """
        generator function to find files in self.path, that end with self.ending
        """
        for f in os.listdir(self.path):
            if len(f) > 4:
                if f[-4:] == self.ending:
                    yield f


#TODO Remove
if __name__ == "__main__":
    parser = FunctionParser('../../../../scripts/builtin')
    path = parser.path + 'kmeans.dml'
    #print(parser.find_header_input_params(path))
    #print(parser.find_function_definition(path))
    print(parser.parse_function(path))
