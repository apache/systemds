#!/usr/bin/env python3
import sys
import os
import subprocess
import argparse

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline

from map_pipeline import SklearnToDMLMapper
from tests.util import test_script, compare_script, suppress_stdout

def test001():
    pipeline = make_pipeline(StandardScaler(), KMeans())
    mapper = SklearnToDMLMapper(pipeline)
    mapper.transform()
    path = 'test001_gen.dml'
    mapper.save(path)
    result = compare_script(path, 'test001.dml')
    if result == True:
        return test_script(path)
    return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='Enable verbose test output.')

    options = parser.parse_args()
    tests = [
        test001
    ]

    results = []

    for test in tests:
        if not options.verbose:
            with suppress_stdout():
                result = test()
        else:
            result = test001()
        results.append(result)
    
    for (t, r) in zip(tests, results):
        print('{}: {}'.format(t.__name__, 'Failed' if not r else 'Success'))
