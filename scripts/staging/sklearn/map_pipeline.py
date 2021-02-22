#!/usr/bin/env python3
import pickle
import sys
import inspect
import mappers

# TODO: rename scirpt, move mappers into won file?,
# own file for each mapper?
# managed by directories?
# import into main script?


def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save(path):
    with open(path, 'rb') as f:
        return pickle.dump(f)


def get_functions():
    clsmembers = inspect.getmembers(sys.modules['mappers'], inspect.isclass)
    functions = {}
    for cls in clsmembers:
        instance = cls[1]()
        if instance.sklearn_name is not None:
            functions[instance.sklearn_name] = cls[1]

    return functions


class SklearnToDMLMapper:
    def __init__(self, pipeline, standalone_script=True): # TODO: standaline_script???
        self.steps = pipeline.steps
        self.functions = get_functions()
        self.dml_script = None
        get_functions()

    def transform(self):
        sources = []
        calls = []

        for i, (name, step) in enumerate(self.steps):
            if name not in self.functions:
                continue

            mapper = self.functions[name](step.get_params())
            calls.append(mapper.get_call())
            sources.append(mapper.get_source())

        self.dml_script = '\n'.join(sources)
        self.dml_script += '\n\n'
        self.dml_script += self.get_input() + '\n'
        self.dml_script += '\n'.join(calls)
        self.dml_script += '\n' + self.get_output()

        return self.dml_script

    def save(self, path):
        if self.dml_script is None:  # TODO handle better? Warning?
            raise RuntimeError('Transformation was not applied yet.')

        with open(path, 'w') as f:
            f.write(self.dml_script)

    def get_input(self):
        func = self.functions[self.steps[0][0]]()
        if func is None:
            raise RuntimeError(f'{self.steps[0][0]} is not supported.')

        if func.is_supervised:
            return 'X = read($X)\nY = read($Y)'
        else:
            return 'X = read($X)'

    def get_output(self):
        func = self.functions[self.steps[-1][0]]()
        if func is None:
            raise RuntimeError(f'{self.steps[-1][0]} is not supported.')
        return f'print({", ".join(func.mapped_output)})'


if __name__ == '__main__':
    pipeline = load('pipe.pkl')

    # TODO internal loading + saving
    mapper = SklearnToDMLMapper(pipeline)
    mapper.transform()
    mapper.save('pipeline.dml')
