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


import json
import os
import re
import textwrap

class FunctionParser(object):
    header_input_pattern = r"^[ \t\n]*[#]+[ \t\n]*input[ \t\n\w:;.,#]*[\s#\-]*[#]+[\w\s\d:,.()\" \t\n\-]*[\s#\-]*$"
    header_output_pattern = r"[\s#\-]*[#]+[ \t]*(return|output)[ \t\w:;.,#]*[\s#\-]*[#]+[\w\s\d:,.()\" \t\-]*[\s#\-]*$"
    function_pattern = r"^[fms]_[\w]+[ \t\n]*=[ \t\n]+function[^#{]*"

    # parameter_pattern = r"^m_[\w]+[\s]+=[\s]+function[\s]*\([\s]*(?=return)[\s]*\)[\s]*return[\s]*\([\s]*([\w\[\]\s,\d=.\-_]*)[\s]*\)[\s]*"
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
        try:
            function_definition = self.find_function_definition(path)
        except AttributeError:
            print(f"[INFO] Skipping '{function_name}': does not match function name pattern. It is likely an internal function.")
            return

        func_split = function_definition.split("function", 1)[1].split("return")

        param_str = self.extract_param_str(func_split[0])
        retval_str = None
        if(len(func_split)> 1):     
            retval_str = self.extract_param_str(func_split[1])

        if param_str:
            parameters = self.get_parameters(param_str)
            return_values = self.get_parameters(retval_str)
            data = {'function_name': function_name,
                    'parameters': parameters, 'return_values': return_values}
            if parameters:
                return data
            else:
                raise AttributeError("Unable to match to function   definition:\n" + function_definition +
                                         "\n parameter_str: " + param_str + "\n     retVal: " + retval_str)
        else:
            raise AttributeError("Unable to match to function definition:\n" +  function_definition +
                                 "\n parameter_str: " + param_str + "\n     retVal: " + retval_str)
   
    def extract_param_str(self, a: str):
        try:
            return a[a.index("(") + 1: a.rindex(")")]
        except:
            raise AttributeError("failed extracting from: " + a)

    def get_parameters(self, param_str: str):
        if(param_str == None):
            return None

        params = re.split(r",[\s]*", param_str)

        paramsCombined = []
        inside = 0

        for param in params:
            before = inside
            start = param.count("(")
            end = param.count(")")
            inside += start - end
            if before > 0:
                if inside > 0:
                    paramsCombined[-1] += param + ","
                else:
                    paramsCombined[-1] += param + ","
            else:
                paramsCombined.append(param)

        parameters = []

        for param in paramsCombined:
            parameters.append(self.parse_single_parameter(param.strip()))
        return parameters

    def parse_single_parameter(self, param: str):
        # try:
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
            if default_value is None:
                raise AttributeError("Failed parsing " + param)

        if "(" in name or "=" in name or "]" in name or "=" in dml_type:
            raise AttributeError("failed Parsing " +
                                 param + "  " + str(splitted))
        return [name, dml_type, default_value]
        # except Exception as e:
        #     import generator
        #     raise AttributeError("Failed parsing " + param + " " + generator.format_exception(e))

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

        description = ""
        h_input = ""
        h_output = ""
        in_input = False
        in_output = False
        with open(path, 'r') as f:
            for _ in range(22):
                line = f.readline()
            while line[0] == '#':
                if "# INPUT:" in line:
                    in_input = True
                    # skip two lines
                    line = f.readline()
                    line = f.readline()
                elif "# OUTPUT:" in line:
                    in_input = False
                    in_output = True
                    # skip two lines
                    line = f.readline()
                    line = f.readline()
                
                if in_output:
                    if "----------" not in line:
                        h_output += line[1:]
                elif in_input:
                    if "----------" not in line:
                        h_input += line[1:]
                else:
                    description += line[1:]
                line = f.readline()

        if description == "" or h_input == "" or h_output == "":
            file_name = os.path.basename(path)
            print("[WARNING] Could not parse header in file \'{file_name}\'.".format(
                file_name=file_name))
            input_parameters = []
            output_parameters = []
        else:
            input_parameters = self.parse_input_output_string(h_input)
            output_parameters = self.parse_input_output_string(h_output)
       
        code_block = None
        with open(path, 'r') as f:
            content = f.read()
            match = re.search(r"#\s*\.\. code-block:: python.*?(?:#\s*-+\n)?(.*?)(?=\n\s*m_\w+\s*= function)", content, re.S)
            if match:
                raw_block = match.group(1)
                # Remove leading `#`
                code_lines = [line.lstrip("#") for line in raw_block.splitlines()]
                code_block = textwrap.dedent("\n".join(code_lines))

        data = {'description': description,
                'parameters': input_parameters,
                'return_values': output_parameters,
                'code_block': code_block}
        return data

    def parse_input_output_string(self, data: str):
        """
            parse the data into a list of tuples containing
            a parameter and a description
        """
        ret = []
        for line in data.split("\n"):
            if line:
                if line[1] == " ":
                    prev = ret[-1]
                    n = (prev[0], prev[1] +"\n        " + line.strip())
                    ret[-1] = n
                    # ret[-1][1] += line.strip()
                else:
                    vd = line.split("  ", 1)
                    ret.append((vd[0].strip(),vd[1].strip()))                
        
        return ret



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
        # if header_param_names != data_param_names:
        # print("[WARNING] The parameter names of the function does not match with the documentation "
        #   "for file \'{file_name}\'.".format(file_name=data["function_name"]))

        header_param_type = [p[1].lower() for p in header["parameters"]]
        header_param_type = [type_mapping["type"].get(
            item, item) for item in header_param_type]

        data_param_type = [p[1].lower() for p in data["parameters"]]
        data_param_type = [type_mapping["type"].get(
            re.search(type_mapping_pattern, str(item).lower()).group() if item else str(item).lower(), item)
            for item in data_param_type]

        # if header_param_type != data_param_type:
        #     print("[WARNING] The parameter type of the function does not match with the documentation "
        #           "for file \'{file_name}\'.".format(file_name=data["function_name"]))
