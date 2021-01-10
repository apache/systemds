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

from typing import Tuple, List
import json
import os
from parser import FunctionParser

class PythonAPIFileGenerator(object):

    target_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'systemds', 'operator')
    source_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))),'scripts', 'builtin')

    def __init__(self):
        super(PythonAPIFileGenerator, self).__init__()

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

    api_template = u"""def {function_name}({parameters}) -> OperationNode:
    {header}
    {value_checks}
    {params_dict}
    return {api_call}\n\n
    """

    kwargs_parameter_string = u"**kwargs: Dict[str, VALID_INPUT_TYPES]"
    kwargs_result = u"params_dict.update(kwargs)"

    value_check_template = u"\n    {param}._check_matrix_op()"
    type_mapping_file = os.path.join('resources','type_mapping.json')

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
        parameters = self.format_param_string(data['parameters'])
        function_name = data['function_name']
        header = data['function_header'] if data['function_header'] else ""
        value_checks = self.format_value_checks(data['parameters'])
        params_dict = self.format_params_dict_string(data['parameters'])
        api_call = self.format_api_call(
            data['parameters'],
            data['return_values'],
            data['function_name']
        )
        #print(parameters)
        #print(function_name)
        #print(header)
        #print(value_checks)
        #print(params_dict)
        #print(api_call)
        return self.__class__.api_template.format(
            function_name=function_name,
            parameters=parameters,
            header=header,
            value_checks=value_checks,
            params_dict=params_dict,
            api_call=api_call
        )
    
    #TODO: mapping of parameter types
    def format_param_string(self, parameters: List[Tuple[str]]) -> str:
        result = u""
        has_optional = False
        path = os.path.dirname(__file__)
        type_mapping_path = os.path.join(path,self.__class__.type_mapping_file)
        #print(type_mapping_path)
        with open(type_mapping_path, 'r') as mapping:
            type_mapping = json.load(mapping)
        #print(type_mapping)
        for param in parameters:
            # map data types
            # TODO: type mapping path
            #param[1] = type_mapping["type"][param[1].lower()]
            param = tuple([type_mapping["type"].get(str(item).lower(),item) for item in param])
            if param[2] is not None:
                has_optional = True
            else:
                if len(result):
                    result = u"{result}, ".format(result=result)
                result = u"{result}{name}: {typ}".format(
                    result=result,
                    name=param[0],
                    typ=param[1]
                )
        if has_optional:
            if len(result):
                result = u"{result}, ".format(result=result)
            result = u"{result}{kwargs}".format(
                result=result,
                kwargs=self.__class__.kwargs_parameter_string
            )
        return result

    def format_params_dict_string(self, parameters: List[Tuple[str]]) -> str:
        if not len(parameters):
            return ""
        has_optional = False
        result = ""
        for param in parameters:
            if param[2] is not None:
                has_optional = True
            else:
                if len(result):
                    result = u"{result}, ".format(
                        result=result)
                else:
                    result = u"params_dict = {"
                result = u"{result}\'{name}\':{name}".format(
                    result=result,
                    name=param[0]
                )
        result = u"{result}}}".format(result=result)
        if has_optional:
            result = u"{result}\n    {kwargs}".format(
                result=result,
                kwargs=self.__class__.kwargs_result
            )
        return result


    #TODO: shape parameter, mapping of return type
    def format_api_call(
        self,
        parameters :List[Tuple[str]],
        return_values :List[Tuple[str]],
        function_name :str
        ) -> str:
        length = len(return_values)
        result = "OperationNode({params})"
        param_string = ""
        param = parameters[0]
        if length > 1:
            output_type_list = ""
            for value in return_values:
                if len(output_type_list):
                    output_type_list = "{output_type_list}, ".format(
                        output_type_list=output_type_list
                    )
                else:
                    output_type_list = "output_types=["
                
                output_type_list = "{output_type_list}OutputType.{typ}".format(
                    output_type_list=output_type_list,
                    typ=value[1]
                )
            output_type_list = "{output_type_list}]".format(
                output_type_list=output_type_list
            )
            output_type = "LIST, number_of_outputs={n}, {output_type_list}".format(
                n=length,
                output_type_list=output_type_list
            )
        else:
            output_type = return_values[0][1]
        result = "{param}.sds_context, \'{function_name}\', named_input_nodes=params_dict, output_type=OutputType.{output_type}".format(
            param=param[0],
            function_name=function_name,
            output_type=output_type
        )
        result ="OperationNode({params})".format(params=result)
        return result

    def format_value_checks(self, parameters :List[Tuple[str]]) -> str:
        result = ""
        for param in parameters:
            if "Matrix" not in param[1]:
                # This check is only needed for Matrix types
                continue
            check = self.__class__.value_check_template.format(param=param[0])
            result = "{result}{check}".format(
                result=result,
                check=check
            )
        return result
    

class PythonAPIDocumentationGenerator(object):
    python_multi_cmnt = "\"\"\""
    param_str = "\n    :param {pname}: {meaning}"
    return_str = "\n    :return: \'OperationNode\' containing {meaning} \n"
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
        input_param = self.header_parameter_string(data["parameters"])
        output_param = self.header_return_string(data["return_values"])

        return self.__class__.python_multi_cmnt + input_param + output_param + "    " + self.__class__.python_multi_cmnt

    def header_parameter_string(self, parameter:dict) -> str:
        parameter_str = ""
        for param in parameter:
            parameter_str += self.__class__.param_str.format(pname=param[0], meaning=param[3])

        return parameter_str

    def header_return_string(self, parameter:dict) -> str:
        meaning_str = ""

        for param in parameter:
            if len(meaning_str) > 0:
                meaning_str += " & " + param[3]
            else:
                meaning_str += param[3]

        return self.__class__.return_str.format(meaning=meaning_str.lower())


#TODO: remove
if __name__ == "__main__":
    f_parser = FunctionParser(PythonAPIFileGenerator.source_path)
    doc_generator = PythonAPIDocumentationGenerator()
    fun_generator = PythonAPIFunctionGenerator()
    for dml_file in f_parser.files():
        header_data = f_parser.parse_header(dml_file)
        try:
            data = f_parser.parse_function(dml_file)
        except AttributeError as e:
            #print("[WARNING] Skipping file \'{file_name}\'.".format(file_name = dml_file))
            continue
        data['function_header'] = doc_generator.generate_documentation(header_data)
        script_content = fun_generator.generate_function(data)
        #print("---------------------------------------")
        #print(script_content)

    """  
    generator = PythonAPIFunctionGenerator()
    data = {'function_header': "\"\"\"\n    sample header\n    \"\"\"",'function_name': 'kmeans', 'parameters': [('X', 'Matrix[Double]', None), ('k', 'Integer', '10'), ('runs', 'Integer', '10'), ('max_iter', 'Integer', '1000'), ('eps', 'Double', '0.000001'), ('is_verbose', 'Boolean', 'FALSE'), ('avg_sample_size_per_centroid', 'Integer', '50'), ('seed', 'Integer', '-1')], 'return_values': [('C', 'Matrix[Double]', None), ('Y', 'Matrix[Double]', None)]}
    document_generator = 
    header = {'function_name': None, 'parameters': [('X', 'Double', '---', 'The input Matrix to do KMeans on.'), ('k', 'Int', '---', 'Number of centroids'), ('runs', 'Int', '10', 'Number of runs (with different initial centroids)'), ('max_iter', 'Int', '1000', 'Maximum number of iterations per run'), ('eps', 'Double', '0.000001', 'Tolerance (epsilon) for WCSS change ratio'), ('is_verbose', 'Boolean', 'FALSE', 'do not print per-iteration stats'), ('avg_sample_size_per_centroid', 'Int', '50', 'Average number of records per centroid in data samples'), ('seed', 'Int', '-1', 'The seed used for initial sampling. If set to -1 random seeds are selected.')], 'return_values': [('Y', 'String', '"Y.mtx"', 'The mapping of records to centroids'), ('C', 'String', '"C.mtx"', 'The output matrix with the centroids')]}
    function_header = document_generator.generate_documentation(header)
    # print(function_header)

    data["function_header"] = function_header
    script_content = generator.generate_function(data)
    print("---------------------------------------")
    print(script_content)"""


