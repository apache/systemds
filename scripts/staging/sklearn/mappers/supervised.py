from .mapper import Mapper


class LinearSVMMapper(Mapper):
    name = 'lv2svm'
    is_supervised = True
    mapped_output = [
        'model'
    ]

    def map_params(self):
        self.mapped_params = [
            'TRUE' if self.params.get('fit_intercept', False) else 'FALSE',
            self.params.get('tol', 0.001),
            self.params.get('C', 1.0),
            self.params.get('max_iter', 100),
            'TRUE' if self.params.get('verbose', False) else 'FALSE',
            -1  # column_id is unkown in sklearn
        ]


class TweedieRegressorMapper(Mapper):
    name = 'glm'
    is_supervised = True
    mapped_output = [
        'beta'
    ]

    def map_params(self):
        # TODO: many parameters cannot be mapped directly:
        # how to handle defaults for dml?
        self.mapped_params = [
            1,  # sklearn impl supports power only, dfam
            self.params.get('power', 0.0),  # vpow
            0,  # link
            1.0,  # lpow
            0.0,  # yneg
            # sklearn does not know last case
            0 if self.params.get('fit_intercept', 1) else 1,
            0.0,  # reg
            self.params.get('tol', 0.000001),
            0.0,  # disp
            200,  # moi
            0  # mii
        ]


class LogisticRegressionMapper(Mapper):
    name = 'multiLogReg'
    is_supervised = True
    mapped_output = [
        'beta'
    ]

    def map_params(self):
        self.mapped_params = [
            # sklearn does not know last case
            0 if self.params.get('fit_intercept', 1) else 1,
            self.params.get('C', 0.0),
            self.params.get('tol', 0.000001),
            100,  # maxi
            0,  # maxii
        ]
