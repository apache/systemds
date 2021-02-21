#!/usr/bin/env python3
import sys
import os
import subprocess
import difflib

def get_systemds_root():
    try:
        return os.environ['SYSTEMDS_ROOT']
    except KeyError as error:
        raise KeyError("SYSTEMDS_ROOT is not set.")
        
def get_sklearn_root():
    return f'{get_systemds_root()}/scripts/staging/sklearn'

# Taken from http://thesmithfam.org/blog/2012/10/25/temporarily-suppress-console-output-in-python/
# Used to suppress intermediate ouput in run_tests.py (verbosity)
from contextlib import contextmanager
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout


def invoke_systemds(path):
    root = get_systemds_root()

    try:
        script_path = os.path.relpath(path, os.getcwd())
        result = subprocess.run([root + "/bin/systemds", script_path],
                             check=True,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             timeout=10000)

    except subprocess.CalledProcessError as systemds_error:
        print("Failed to run systemds!")
        print("Error code: " + str(systemds_error.returncode))
        print("Stdout:")
        print(systemds_error.output.decode("utf-8"))
        print("Stderr:")
        print(systemds_error.stderr.decode("utf-8"))
        return False
    return True

def test_script(path):
    print('#' * 30)
    print('Running generated script on systemds.')
    result = invoke_systemds(path)
    print('Finished test.')
    return result

def compare_script(actual, expected):
    try:
        f_expected = open(f'{get_sklearn_root()}/tests/expected/{expected}')
        f_actual = open(f'{get_sklearn_root()}/{actual}')
        diff = difflib.ndiff(f_actual.readlines(), f_expected.readlines())
        changes = [l for l in diff if not l.startswith('  ')]
        print('#' * 30)
        if len(changes) == 0:
            print('Actual script matches expected script.')
            return True
        else:
            print('Actual script does not match expected script.')
            print('Legend:')
            print('    "+ " ... line unique to actual script')
            print('    "- " ... line unique to expected script')
            print('    "? " ... linue not present in either script')
            print('#' * 30)
            print(*changes)
            print('#' * 30)
            return False

    except Exception as e:
        print('Failed to compare script.')
        print(e)
        return False