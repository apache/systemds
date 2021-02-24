#!/usr/bin/env python3
import pickle
import sys
import inspect
import mappers
import argparse

class SklearnToDMLMapper:
    def __init__(self, pipeline, input_name='input'):
        self.steps = pipeline.steps
        self.functions = self.__get_functions()
        self.dml_script = None
        self.input_name = input_name

    def __get_functions(self):
        clsmembers = inspect.getmembers(sys.modules['mappers'], inspect.isclass)
        functions = {}
        for cls in clsmembers:
            instance = cls[1]()
            if instance.sklearn_name is not None:
                functions[instance.sklearn_name] = cls[1]

        return functions

    def __get_input(self):
        # Get last function (an algorithm)
        func = self.functions[self.steps[-1][0]]()
        if func is None:
            raise RuntimeError(f'{self.steps[-1][0]} is not supported.')

        if func.is_supervised:
            return f'X = read(${self.input_name}_X)\nY = read(${self.input_name}_Y)'
        else:
            return f'X = read(${self.input_name}_X)'

    def __get_output(self):
        func = self.functions[self.steps[-1][0]]()
        if func is None:
            raise RuntimeError(f'{self.steps[-1][0]} is not supported.')
        return '\n'.join([f'write({output}, "{output}.csv")' for output in func.mapped_output])

    def transform(self):
        sources = []
        calls = []

        for name, step in self.steps:
            if name not in self.functions:
                continue

            mapper = self.functions[name](step.get_params())
            calls.append(mapper.get_call())
            sources.append(mapper.get_source())

        self.dml_script = "{}\n\n{}\n\n{}\n\n{}".format('\n'.join(sources), 
                                                        self.__get_input(), 
                                                        '\n'.join(calls), 
                                                        self.__get_output())
        return self.dml_script

    def save(self, path):
        if self.dml_script is None:
            raise RuntimeError('Transformation was not applied yet.')

        with open(path, 'w') as f:
            f.write(self.dml_script)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tool that parses a sklearn pipeline and produces a dml script')
    parser.add_argument('Path',
            metavar='path',
            type=str,
            help='location of the sklearn pipline pickle file')
    parser.add_argument('-i',
            metavar='input_name',
            type=str,
            default='X',
            help='name for the input variable')
    parser.add_argument('-o',
            metavar='output',
            type=str,
            default='./pipeline.dml',
            help='path for the dml output script')

    args = parser.parse_args()

    try:
        with open(args['path'], 'rb') as f:
            pipeline = pickle.load(f)

        mapper = SklearnToDMLMapper(pipeline, args['input_name'])
        mapper.transform()
        mapper.save(args['output'])
    except Exception as e:
        print(f'Failed to transform pipeline.\nError:\n{e}')