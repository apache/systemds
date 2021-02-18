#!/usr/bin/env python3
import pickle
import os

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

def scripts_home():
    systemds_home = os.getenv('SYSTEMDS_HOME')
    if systemds_home is None:
        return builtin_path
    else:
        return f'{systemds_home}/{builtin_path}'

# TODO: path handling with Path or env variablebuiltin_path
builtin_path = "scripts/builtin"

class Mapper:
    def __init__(self):
        self.name = None
        self.mapped_params = []
        self.mapped_output = []
        self.is_intermediate = None
        self.is_supervised = False # TODO: default cases. Better if set by own Mappers?

    def get_source(self):
        return 'source("{}/{}") as ns_{}'.format(scripts_home(), self.name, self.name) 

    # TODO better string building
    def get_call(self):
        # TODO: handle intermediate step results
        input_ = 'X, y' if self.is_supervised else 'X'
        output_ = ', '.join(self.mapped_output) if not self.is_intermediate else 'X'
        param_ = ', '.join(map(str, self.mapped_params))
        call = "{} = ns_{}::m_{}({}, {})".format(output_, self.name, self.name, input_, param_)
        return call

    def __map_parameters(self, params):
        raise NotImplementedError('Base class is not implemented.')

    def __map_output(self):
        raise NotImplementedError('Base class is not implemented.')

# TODO: missing parameter mapping
class KmeansMapper(Mapper):
    def __init__(self):
        super().__init__()
        self.name = 'kmeans'
        self.is_intermediate = False
        self.is_supervised = False
    
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
        super().__init__()
        self.name = 'dbscan'
        self.is_intermediate = False
        self.is_supervised = False
    
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
        super().__init__()
        self.name = 'l2svm' # Handle model validation?
        self.is_intermediate = False
        self.is_supervised = True
    
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
        super().__init__()
        self.name = 'gmm'
        self.is_intermediate = False
        self.is_supervised = False
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

# GMM Mapper
class TweedieRegressorMapper(Mapper):
    def __init__(self):
        super().__init__()
        self.name = 'glm'
        self.is_intermediate = False
        self.is_supervised = True
    
    def get_call(self, parameters):
        self.__map_parameters(parameters)
        self.__map_output()
        return super().get_call()

    def __map_parameters(self, params):
        # TODO: many parameters cannot be mapped directly: how to handle defaults for dml?
        self.mapped_params = [
            1, # sklearn impl supports power only, dfam
            params.get('power', 0.0), # vpow
            0, # link
            1.0, # lpow
            0.0, #yneg
            0 if params.get('fit_intercept', 1) else 1, # sklearn does not know last case
            0.0, # reg
            params.get('tol', 0.000001),
            0.0, # disp
            200, # moi
            0 # mii
        ]

    def __map_output(self):
        self.mapped_output = [
            'beta'
        ]

# multiLogMapper
class LogisticRegressionMapper(Mapper):
    def __init__(self):
        super().__init__()
        self.name = 'multiLogReg'
        self.is_intermediate = False
        self.is_supervised = True
    
    def get_call(self, parameters):
        self.__map_parameters(parameters)
        self.__map_output()
        return super().get_call()

    def __map_parameters(self, params):
        self.mapped_params = [
            0 if params.get('fit_intercept', 1) else 1, # sklearn does not know last case
            params.get('C', 0.0),
            params.get('tol', 0.000001),
            100, # maxi
            0, # maxii
        ]

    def __map_output(self):
        self.mapped_output = [
            'beta'
        ]

class StandardScalerMapper(Mapper):
    def __init__(self):
        super().__init__()
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
        super().__init__()
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

class SimpleImputerMapper(Mapper):
    def __init__(self):
        super().__init__()
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

# Split is not actually part of a pipeline
# TODO: Remove (out of scope?) 
class SplitMapper(Mapper):
    def __init__(self):
        super().__init__()
        self.name = 'split'
        self.is_intermediate = True

    def get_call(self, parameters):
        self.__map_parameters(parameters)
        self.__map_output()
        return super().get_call()

    def __map_parameters(self, params):
        self.mapped_params = [
            params.get('train_size', 0.7),
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

# TODO: can be pipelined in sklearn, but how in dml?
class PCAMapper(Mapper):
    def __init__(self):
        super().__init__()
        self.name = 'pca'
        self.is_intermediate = True

    def get_call(self, parameters):
        self.__map_parameters(parameters)
        self.__map_output()
        return super().get_call()

    def __map_parameters(self, params):
        self.mapped_params = [
            params.get('n_components'),
            1,   #non existant in SKlearn
            1    #non existant in SKlearn
        ]

    def __map_output(self):
        self.mapped_output = [
            'Xout',
            'Mout'
        ]

# TODO: can scores be added to the end of a pipeline?
class distMapper(Mapper):
    def __init__(self):
        super().__init__()
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
        super().__init__()
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
            calls.append(mapper.get_call(step.get_params()))
            sources.append(mapper.get_source())

        self.dml_script = '\n'.join(sources)
        self.dml_script += '\n\n'
        self.dml_script += self.get_input() + '\n'
        self.dml_script += '\n'.join(calls)
        self.dml_script += '\n' + self.get_output()

        print('Mapped DML script:')
        print(self.dml_script)

    def save(self, path):
        if self.dml_script == None: # TODO handle better? Warning?
            raise RuntimeError('Transformation was not applied yet.')

        with open(path, 'w') as f:
            f.write(self.dml_script)

    def get_input(self):
        if self.steps[0].is_intermediate:
            return 'X = read($X)\nY = read($Y)'
        else:
            return 'X = read($X)'

    def get_output(self):
        return 'print(X)'

if __name__ == '__main__':
    pipeline = load('pipe.pkl')

    # TODO internal loading + saving
    mapper = SklearnToDMLMapper(pipeline)
    mapper.transform()
    mapper.save('pipeline.dml')

