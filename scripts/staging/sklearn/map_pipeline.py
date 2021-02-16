#!/usr/bin/env python3
import pickle

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

class DBSCANMapper(Mapper):
    def __init__(self):
        self.name = 'dbscan'
        self.is_intermediate = False
    
    def get_call(self, parameters):
        self.__map_parameters(parameters)
        self.__map_output()
        return super().get_call()

    def __map_parameters(self, params):
        self.mapped_params = [
            params.get('eps', 0.5),
            params.get('min_samples', 5)
        ]

    def __map_output(self):
        self.mapped_output = [
            'clusterMembers'
        ]

class LinearSVMMapper(Mapper):
    def __init__(self):
        self.name = 'l2svm' # Handle model validation?
        self.is_intermediate = False
    
    def get_call(self, parameters):
        self.__map_parameters(parameters)
        self.__map_output()
        return super().get_call()

    def __map_parameters(self, params):
        self.mapped_params = [
            params.get('fit_intercept', False),
            params.get('tol', 0.001),
            params.get('C', 1.0),
            params.get('max_iter', 100),
            params.get('verbose', False),
            -1 # column_id is unkown in sklearn
        ]

    def __map_output(self):
        self.mapped_output = [
            'model'
        ]

class GaussianMixtureMapper(Mapper):
    def __init__(self):
        self.name = 'gmm'
        self.is_intermediate = False
        self.model_map = {
            'full': 'VVV',
            'tied': 'EEE',
            'diag': 'VVI',
            'spherical': 'VVI'
        }
    
    def get_call(self, parameters):
        self.__map_parameters(parameters)
        self.__map_output()
        return super().get_call()

    def __map_parameters(self, params):
        self.mapped_params = [
            params.get('n_components', 3),
            self.model_map.get(params.get('covariance_type', 'VVV')),
            params.get('init_params', 'kmeans'),
            params.get('max_iter', 100),
            params.get('reg_covar', 1e-6),
            params.get('tol', 0.000001)
        ]

    def __map_output(self):
        self.mapped_output = [
            'weight',
            'labels',
            'df',
            'bic'
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

