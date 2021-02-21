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

def test002():
    pipeline = make_pipeline(normalize(), KMeans())
    mapper = SklearnToDMLMapper(pipeline)
    mapper.transform()
    path = 'test002_gen.dml'
    mapper.save(path)
    result = compare_script(path, 'test002.dml')
    if result == True:
        return test_script(path)
    return False

def test003():
    pipeline = make_pipeline(imputeByMean(), KMeans())
    mapper = SklearnToDMLMapper(pipeline)
    mapper.transform()
    path = 'test003_gen.dml'
    mapper.save(path)
    result = compare_script(path, 'test003.dml')
    if result == True:
        return test_script(path)
    return False

def test004():
    pipeline = make_pipeline(imputByMedian(), KMeans())
    mapper = SklearnToDMLMapper(pipeline)
    mapper.transform()
    path = 'test004_gen.dml'
    mapper.save(path)
    result = compare_script(path, 'test004.dml')
    if result == True:
        return test_script(path)
    return False

def test005():
    pipeline = make_pipeline(pca(), KMeans())
    mapper = SklearnToDMLMapper(pipeline)
    mapper.transform()
    path = 'test005_gen.dml'
    mapper.save(path)
    result = compare_script(path, 'test005.dml')
    if result == True:
        return test_script(path)
    return False

def test006():
    pipeline = make_pipeline(StandardScaler(), dbscan())
    mapper = SklearnToDMLMapper(pipeline)
    mapper.transform()
    path = 'test006_gen.dml'
    mapper.save(path)
    result = compare_script(path, 'test006.dml')
    if result == True:
        return test_script(path)
    return False

def test007():
    pipeline = make_pipeline(normalize(), dbscan())
    mapper = SklearnToDMLMapper(pipeline)
    mapper.transform()
    path = 'test007_gen.dml'
    mapper.save(path)
    result = compare_script(path, 'test007.dml')
    if result == True:
        return test_script(path)
    return False

def test008():
    pipeline = make_pipeline(imputeByMean(), dbscan())
    mapper = SklearnToDMLMapper(pipeline)
    mapper.transform()
    path = 'test008_gen.dml'
    mapper.save(path)
    result = compare_script(path, 'test008.dml')
    if result == True:
        return test_script(path)
    return False

def test009():
    pipeline = make_pipeline(imputeByMedian(), dbscan())
    mapper = SklearnToDMLMapper(pipeline)
    mapper.transform()
    path = 'test009_gen.dml'
    mapper.save(path)
    result = compare_script(path, 'test009.dml')
    if result == True:
        return test_script(path)
    return False

def test010():
    pipeline = make_pipeline(pca(), dbscan())
    mapper = SklearnToDMLMapper(pipeline)
    mapper.transform()
    path = 'test010_gen.dml'
    mapper.save(path)
    result = compare_script(path, 'test010.dml')
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
        test002
        test003
        test004
        test005
        test006
        test007
        test008
        test009
        test010

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
