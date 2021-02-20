from .mapper import Mapper


class DistMapper(Mapper):
    name = 'dist'
    mapped_output = [
        'X'
    ]

    def map_params(self):
        self.mapped_params = []


class AccuracyMapper(Mapper):
    name = 'getAccuracy'
    mapped_output = [
        'accuracy'
    ]

    def map_params(self):
        self.mapped_params = [
            1
        ]
