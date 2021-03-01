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
import json


class FunctionParser(object):
    header_input_pattern = r"^[ \t\n]*[#]+[ \t\n]*input[ \t\n\w:;.,#]*[\s#\-]*[#]+[\w\s\d:,.()\" \t\n\-]*[\s#\-]*$"
    header_output_pattern = r"[\s#\-]*[#]+[ \t]*(return|output)[ \t\w:;.,#]*[\s#\-]*[#]+[\w\s\d:,.()\" \t\-]*[\s#\-]*$"
    function_pattern = r"^m_[\w]+[ \t\n]+=[ \t\n]+function[^#{]*"
    parameter_pattern = r"^m_[\w]+[\s]+=[\s]+function[\s]*\([\s]*([\w\[\]\s,\d=.\-'\"_\.]*)[\s]*\)[\s]*return[\s]*\([\s]*([\w\[\]\s,\d=.\-_]*)[\s]*\)[\s]*"
    header_parameter_pattern = r"[\s#\-]*[#]+[ \t]*([\w|-]+)[\s]+([\w]+)[\s]+([\w,\d.\"\-]+)[\s]+([\w|\W]+)"
    divider_pattern = r"[\s#\-]*"

    type_mapping_file = os.path.join('resources', 'type_mapping.json')

    def __init__(self, path: str, extension: str = 'dml'):
        """
        @param path: path where to look for python scripts
        """
        super(FunctionParser, self).__init__()
        self.path = path
        self.extension = '.{extension}'.format(extension=extension)
        self.files()

    def parse_function(self, path: str):
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
        file_name = os.path.basename(path)
        function_name, extension = os.path.splitext(file_name)
        function_definition = self.find_function_definition(path)
        pattern = re.compile(
            self.__class__.parameter_pattern, flags=re.I | re.M)
        match = pattern.match(function_definition)
        if match:
            param_str, retval_str = match.group(1, 2)
            parameters = self.get_parameters(param_str)
            return_values = self.get_parameters(retval_str)
        else:
            # TODO handle default matrix variables. 
            raise AttributeError("Unable to match to function definition:\n" + function_definition)
        data = {'function_name': function_name,
                    'parameters': parameters, 'return_values': return_values}
        return data

    def get_parameters(self, param_str: str):
        
        params = re.split(r",[\s]*", param_str)
        
        parameters = []
        for param in params:
            parameters.append(self.parse_single_parameter(param))
        return parameters

    def parse_single_parameter(self, param: str):
        splitted = re.split(r"[\s]+", param)
        dml_type = splitted[0]
        name = splitted[1]
        default_value = None

        if len(splitted) == 4:
            if splitted[2] == "=":
                default_value = splitted[3]
        elif "=" in name:
            default_split = name.split("=")
            name = default_split[0]
            default_value = default_split[1]
        return (name, dml_type, default_value)

    def get_header_parameters(self, param_str: str):
        parameters = list()
        pattern = re.compile(
            self.__class__.header_parameter_pattern, flags=re.I)

        for param_line in [s for s in param_str.split("\n") if s]:
            match = pattern.match(param_line)
            try:
                parameters.append((match.group(1), match.group(
                    2), match.group(3), match.group(4)))
            except Exception as e:
                if re.search(pattern=self.__class__.divider_pattern, string=param_line, flags=re.I | re.M) is not None:
                    continue
                print(e)
                return parameters

        return parameters

    def parse_header(self, path: str):
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
        try:
            h_input = self.find_header_input_params(path)
            input_parameters = self.get_header_parameters(h_input)

            h_output = self.find_header_output_params(path)
            output_parameters = self.get_header_parameters(h_output)
        except AttributeError as e:
            file_name = os.path.basename(path)
            print("[WARNING] Could not parse header in file \'{file_name}\'.".format(
                file_name=file_name))
            input_parameters = []
            output_parameters = []
        data = {'function_name': None, 'parameters': input_parameters,
                'return_values': output_parameters}
        return data

    def find_header_input_params(self, path: str):
        with open(path, 'r') as f:
            content = f.read()
        start = re.search(pattern=self.__class__.header_input_pattern,
                          string=content, flags=re.I | re.M).end()
        end = re.search(pattern=self.__class__.header_output_pattern,
                        string=content, flags=re.I | re.M).start()
        header = content[start:end]
        return header

    def find_header_output_params(self, path: str):
        with open(path, 'r') as f:
            content = f.read()
        start = re.search(pattern=self.__class__.header_output_pattern,
                          string=content, flags=re.I | re.M).end()
        end = re.search(pattern=self.__class__.function_pattern,
                        string=content, flags=re.I | re.M).start()
        header = content[start:end]
        return header

    def find_function_definition(self, path: str):
        with open(path, 'r') as f:
            content = f.read()
        match = re.search(pattern=self.__class__.function_pattern,
                          string=content, flags=re.I | re.M)
        if match:
            start = match.start()
            end = match.end()
            return content[start:end]
        else:
            raise AttributeError("Function definition not found in : " + path)

    def files(self):
        """
        generator function to find files in self.path, that end with self.extension
        """
        files = os.listdir(self.path)
        files.sort()
        for f in files:
            name, extension = os.path.splitext(f)
            if extension == self.extension:
                yield os.path.join(self.path, f)

    def check_parameters(self, header, data):
        type_mapping_pattern = r"^([^\[\s]+)"

        path = os.path.dirname(__file__)
        type_mapping_path = os.path.join(
            path, self.__class__.type_mapping_file)

        with open(type_mapping_path, 'r') as mapping:
            type_mapping = json.load(mapping)

        header_param_names = [p[0].lower() for p in header["parameters"]]
        data_param_names = [p[0].lower() for p in data["parameters"]]
        if header_param_names != data_param_names:
            print("[WARNING] The parameter names of the function does not match with the documentation "
                  "for file \'{file_name}\'.".format(file_name=data["function_name"]))

        header_param_type = [p[1].lower() for p in header["parameters"]]
        header_param_type = [type_mapping["type"].get(
            item, item) for item in header_param_type]

        data_param_type = [p[1].lower() for p in data["parameters"]]
        data_param_type = [type_mapping["type"].get(
            re.search(type_mapping_pattern, str(item).lower()).group() if item else str(item).lower(), item)
            for item in data_param_type]

        if header_param_type != data_param_type:
            print("[WARNING] The parameter type of the function does not match with the documentation "
                  "for file \'{file_name}\'.".format(file_name=data["function_name"]))
