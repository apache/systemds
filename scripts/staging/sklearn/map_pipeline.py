#!/usr/bin/env python3
import pickle

def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save(path):
    with open(path, 'rb') as f:
        return pickle.dump(f)

# TODO
builtin_path = "scripts/builtin"

class Mapper:
    def __init__(self):
        self.name = None
        self.mapped_params = []
        self.mapped_output = []
        self.is_intermediate = None
        raise NotImplementedError('Base class is not implemented.')

    def get_source(self):
        return 'source("{}/{}") as ns'.format(builtin_path, self.name) 

    # TODO better string building
    def get_call(self):
        # TODO: handle intermediate step results
        if self.is_intermediate:
            call = 'X'
        else:
            call = ', '.join(self.mapped_output)
        call += ' = ns::m_{}(X'.format(self.name)
        for p in self.mapped_params:
            call += ', '
            call += '{}'.format(p)

        call += ')'
        return call

    def __map_parameters(self, params):
        raise NotImplementedError('Base class is not implemented.')

    def __map_output(self):
        raise NotImplementedError('Base class is not implemented.')

# TODO: missing parameter mapping
class KmeansMapper(Mapper):
    def __init__(self):
        self.name = 'kmeans'
        self.is_intermediate = False
    
    def get_call(self, parameters):
        self.__map_parameters(parameters)
        self.__map_output()
        return super().get_call()

    def __map_parameters(self, params):
        self.mapped_params = [
            params['n_clusters'],
            params['n_init'],
            params['max_iter'],
            params['tol'],
            params.get('verbose', False),
            50, # avg_sample_size_per_centroid unkown in sklearn
            -1 if params['random_state'] == None else params['random_state']  
        ]

    def __map_output(self):
        self.mapped_output = [
            'C', # The output matrix with the centroids
            'Y'  # The mapping of records to centroids
        ]

class SplitMapper(Mapper):
    def __init__(self):
        self.name = 'split'
        self.is_intermediate = True

    def get_call(self, parameters):
        self.__map_parameters(parameters)
        self.__map_output()
        return super().get_call()

    def __map_parameters(self, params):
        self.mapped_params = [
            params['train_size'],
            True,   #cant be done 100% accurate in SKlearn look up later
            -1 if params['random_state'] == None else params['random_state']
        ]

    def __map_output(self):
        self.mapped_output = [
            'Xtrain',
            'Xtest',
            'ytrain',
            'ytest'
        ]

class StandardScalerMapper(Mapper):
    def __init__(self):
        self.name = 'scale'
        self.is_intermediate = True
    
    def get_call(self, parameters):
        self.__map_parameters(parameters)
        self.__map_output()
        return super().get_call()

    def __map_parameters(self, params):
        self.mapped_params = [
            params['with_mean'],
            params['with_std']
        ]

    def __map_output(self):
        self.mapped_output = [
            'Y'
        ]


class NormalizeMapper(Mapper):
    def __init__(self):
        self.name = 'normalize'
        self.is_intermediate = True

    def get_call(self, parameters):
        self.__map_parameters(parameters)
        self.__map_output()
        return super().get_call()

    def __map_parameters(self, params):
        self.mapped_params = [
        ]

    def __map_output(self):
        self.mapped_output = [
            'Y'
        ]

class PCAMapper(Mapper):
    def __init__(self):
        self.name = 'pca'
        self.is_intermediate = True

    def get_call(self, parameters):
        self.__map_parameters(parameters)
        self.__map_output()
        return super().get_call()

    def __map_parameters(self, params):
        self.mapped_params = [
            params.get('n_components'),
            True,   #non existant in SKlearn
            True    #non existant in SKlearn
        ]

    def __map_output(self):
        self.mapped_output = [
            'Xout',
            'Mout'
        ]


class SimpleImputerMapper(Mapper):
    def __init__(self):

        self.is_intermediate = True

    def get_call(self, parameters):
        self.__map_parameters(parameters)
        self.__map_output()
        return super().get_call()

    def __map_parameters(self, params): # might update naming ?
        if params['startegy'] == 'median':
            self.name = 'imputeByMedian'
        else:
            self.name = 'imputeByMean'

        self.mapped_params = [

        ]

    def __map_output(self):
        self.mapped_output = [
            'X'
        ]

class distMapper(Mapper):
    def __init__(self):
        self.name = 'dist'
        self.is_intermediate = False #Edge cases?

    def get_call(self, parameters):
        self.__map_parameters(parameters)
        self.__map_output()
        return super().get_call()

    def __map_parameters(self, params):
        self.mapped_params = [

        ]

    def __map_output(self):
        self.mapped_output = [
            'X'
        ]

class accuracyMapper(Mapper):
    def __init__(self):
        self.name = 'getAccuracy'
        self.is_intermediate = False #Edge cases?

    def get_call(self, parameters):
        self.__map_parameters(parameters)
        self.__map_output()
        return super().get_call()

    def __map_parameters(self, params):
        self.mapped_params = [
            False
        ]

    def __map_output(self):
        self.mapped_output = [
            'accuracy'
        ]


class SklearnToDMLMapper:
    def __init__(self, pipeline):
        self.steps = pipeline.steps
        self.functions = {
            'kmeans': KmeansMapper,
            'standardscaler': StandardScalerMapper
        }
        self.dml_script = None

    def transform(self):
        sources = []
        calls = []

        for i, (name, step) in enumerate(self.steps):
            if name not in self.functions:
                # TODO handle missing function mapping
                continue
            mapper = self.functions[name]()
            sources.append(mapper.get_source())
            calls.append(mapper.get_call(step.get_params()))

        self.dml_script = '\n'.join(sources)
        self.dml_script += '\n\n'
        self.dml_script += '\n'.join(calls)

        print('Mapped DML script:')
        print(self.dml_script)

    def save(self, path):
        if self.dml_script == None: # TODO handle better? Warning?
            raise RuntimeError('Transformation was not applied yet.')

        with open(path, 'w') as f:
            f.write(self.dml_script)

if __name__ == '__main__':
    pipeline = load('pipe.pkl')

    # TODO internal loading + saving
    mapper = SklearnToDMLMapper(pipeline)
    mapper.transform()
    mapper.save('pipeline.dml')

