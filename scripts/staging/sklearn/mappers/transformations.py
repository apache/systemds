from .mapper import Mapper


class StandardScalerMapper(Mapper):
    name = 'scale'
    sklearn_name = 'standardscaler'
    is_intermediate = True
    mapped_output = [
        'Y'
    ]

    def map_params(self):
        self.mapped_params = [
            self.params['with_mean'],
            self.params['with_std']
        ]


class SplitMapper(Mapper):
    name = 'split'
    is_intermediate = True
    mapped_output = [
        'Xtrain',
        'Xtest',
        'ytrain',
        'ytest'
    ]

    def map_params(self):
        self.mapped_params = [
            self.params.get('train_size', 0.7),
            1,  # cant be done 100% accurate in SKlearn look up later
            -1 if self.params['random_state'] is None \
            else self.params['random_state']
        ]


class NormalizeMapper(Mapper):
    name = 'normalize'
    sklearn_name = 'normalizer'
    is_intermediate = True
    mapped_output = [
        'Y'
    ]

    def map_params(self):
        self.mapped_params = []


class SimpleImputerMapper(Mapper):
    name = 'impute'
    sklearn_name = 'simpleimputer'
    is_intermediate = True
    mapped_output = [
        'X'
    ]

    def map_params(self):  # might update naming ?
        if self.params['strategy'] == 'median':
            self.name = 'imputeByMedian'
        else:
            self.name = 'imputeByMean'

        self.mapped_params = []


class PCAMapper(Mapper):
    name = 'pca'
    sklearn_name = 'pca'
    is_intermediate = True
    mapped_output = [
        'Xout',
        'Mout'
    ]

    def map_params(self):
        self.mapped_params = [
            self.params.get('n_components'),
            True,  # non existant in SKlearn
            True  # non existant in SKlearn
        ]
