#!/usr/bin/env python3
import sys
import os
import subprocess
import argparse
import logging

from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.pipeline import make_pipeline

from map_pipeline import SklearnToDMLMapper
from tests.util import test_script, compare_script, get_systemds_root

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
    pipeline = make_pipeline(Normalizer(), KMeans())
    mapper = SklearnToDMLMapper(pipeline)
    mapper.transform()
    path = 'test002_gen.dml'
    mapper.save(path)
    result = compare_script(path, 'test002.dml')
    if result == True:
        return test_script(path)
    return False

def test003():
    pipeline = make_pipeline(SimpleImputer(strategy='mean'), KMeans())
    mapper = SklearnToDMLMapper(pipeline)
    mapper.transform()
    path = 'test003_gen.dml'
    mapper.save(path)
    result = compare_script(path, 'test003.dml')
    if result == True:
        return test_script(path)
    return False

def test004():
    pipeline = make_pipeline(SimpleImputer(strategy='median'), KMeans())
    mapper = SklearnToDMLMapper(pipeline)
    mapper.transform()
    path = 'test004_gen.dml'
    mapper.save(path)
    result = compare_script(path, 'test004.dml')
    if result == True:
        return test_script(path)
    return False

def test005():
    pipeline = make_pipeline(PCA(), KMeans())
    mapper = SklearnToDMLMapper(pipeline)
    mapper.transform()
    path = 'test005_gen.dml'
    mapper.save(path)
    result = compare_script(path, 'test005.dml')
    if result == True:
        return test_script(path)
    return False

def test006():
    pipeline = make_pipeline(StandardScaler(), DBSCAN())
    mapper = SklearnToDMLMapper(pipeline)
    mapper.transform()
    path = 'test006_gen.dml'
    mapper.save(path)
    result = compare_script(path, 'test006.dml')
    if result == True:
        return test_script(path)
    return False

def test007():
    pipeline = make_pipeline(Normalizer(), DBSCAN())
    mapper = SklearnToDMLMapper(pipeline)
    mapper.transform()
    path = 'test007_gen.dml'
    mapper.save(path)
    result = compare_script(path, 'test007.dml')
    if result == True:
        return test_script(path)
    return False

def test008():
    pipeline = make_pipeline(SimpleImputer(strategy='mean'), DBSCAN())
    mapper = SklearnToDMLMapper(pipeline)
    mapper.transform()
    path = 'test008_gen.dml'
    mapper.save(path)
    result = compare_script(path, 'test008.dml')
    if result == True:
        return test_script(path)
    return False

def test009():
    pipeline = make_pipeline(SimpleImputer(strategy='median'), DBSCAN())
    mapper = SklearnToDMLMapper(pipeline)
    mapper.transform()
    path = 'test009_gen.dml'
    mapper.save(path)
    result = compare_script(path, 'test009.dml')
    if result == True:
        return test_script(path)
    return False

def test010():
    pipeline = make_pipeline(PCA(), DBSCAN())
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
    parser.add_argument('--log', action='store', default='ERROR', 
                        help='Set logging level (ERROR, INFO, DEBUG).')

    options = parser.parse_args()
    numeric_level = getattr(logging, options.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {options.log}')
    logging.basicConfig(level=numeric_level)

    tests = [
        test001,
        test002,
        test003,
        test004,
        test005,
        test006,
        test007,
        test008,
        test009,
        test010
    ]

    results = []

    try:
        get_systemds_root()
    except Exception as e:
        logging.error(e)
        exit(-1)

    for test in tests:
        logging.info('*' * 50)
        logging.info((18*'*' + test.__name__ + (50-20-len(test.__name__)) * '*'))
        result = test001()
        results.append(result)
    
    print('*' * 50)
    print('Finished all tests.')
    for (t, r) in zip(tests, results):
        print('{}: {}'.format(t.__name__, 'Failed' if not r else 'Success'))
