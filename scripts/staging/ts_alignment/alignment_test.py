import numpy as np
from lib.alignment_statistics import AlignmentStatistics
from lib.evaluate_alignment import Evaluator

from lib.distance_measure import EuclideanDistance, PearsonCorrelation, CosineSimilarity
from lib.alignment_strategies import Alignment, ChunkedCCRAlignment, CCRAlignment, DynamicTimeWarpingAlignment
import lib.majority_voting as majority_voting
from lib.sensor_data_loader import SensorDataLoader
import unittest
import shutil


class TestAlignment(unittest.TestCase):
    def setUp(self):
        self.x = SensorDataLoader('data/x.csv', 1000, fps=1000)
        self.y = SensorDataLoader('data/y.csv', 1000, x=self.x, fps=1000)

    def test_static_ccr(self):
        ccr_alignment = Alignment(CCRAlignment(self.x, self.y, PearsonCorrelation()))
        lag = ccr_alignment.compute_alignment()

        evaluator = Evaluator(self.x.df, self.y.df, self.x)
        score = evaluator.evaluate(evaluator.parse_lags(lag))

        self.assertEqual(lag, 0.006)
        self.assertEqual(score, 0.3022069060189968)

    def test_chunked_ccr_uv(self):
        score = self.ccr_alignment(True)
        self.assertEqual(score, 0.9984141629904713)

    def test_chunked_ccr_mv(self):
        score = self.ccr_alignment(False)
        self.assertEqual(score, 0.9984141629904713)

    def test_dtw(self):
        dtw_alignment = Alignment(
            DynamicTimeWarpingAlignment(self.x, self.y, False))

        dtw_alignment_path = dtw_alignment.compute_alignment()
        evaluator = Evaluator(self.x.df, self.y.df, self.x)
        score = evaluator.evaluate(np.asarray(dtw_alignment_path))

        self.assertEqual(score, 0.9904705134588262)

    def ccr_alignment(self, is_uv):
        folder_name = "test_chunked_ccr"
        measures = [PearsonCorrelation(), CosineSimilarity(), EuclideanDistance()]
        stats = AlignmentStatistics(measures)
        stats.add_index(1)
        chunked_ccr_alignment_gut = Alignment(
            ChunkedCCRAlignment(self.x, self.y, measures, folder_name, 1,
                                verbose=False, compute_uv=is_uv))
        chunked_ccr_alignment_gut.compute_alignment()

        evaluator = Evaluator(self.x.df, self.y.df, self.x)

        for measure in measures:
            m = majority_voting.MajorityVoting(folder_name, 10 if is_uv else -1, measures, 1)
            lags = m.calculate_majority_lags_for_measure(measure)
            score = evaluator.evaluate(evaluator.parse_lags(lags))
            stats.add_score_for_measure(measure, score)
            stats.extract_stats(lags, measure.name)
        stats.evaluate_best_stats()

        shutil.rmtree(folder_name, ignore_errors=True)

        return stats.max_score


if __name__ == '__main__':
    unittest.main()
