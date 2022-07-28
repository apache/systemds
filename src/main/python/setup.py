#!/usr/bin/env python3
#-------------------------------------------------------------
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
#-------------------------------------------------------------

import sys
from setuptools import find_packages, setup
import time

try:
    exec(open('systemds/project_info.py').read())
except IOError:
    print("Could not read project_info.py. Will use default values.", file=sys.stderr)
    BUILD_DATE_TIME = str(time.strftime("%Y%m%d.%H%M%S"))
    __project_artifact_id__ = 'systemds'
    __project_version__ = BUILD_DATE_TIME + '.dev0'
ARTIFACT_NAME = __project_artifact_id__
ARTIFACT_VERSION = __project_version__
ARTIFACT_VERSION_SHORT = ARTIFACT_VERSION.split("-")[0]

REQUIRED_PACKAGES = [
    'numpy >= 1.8.2',
    'py4j >= 0.10.9',
    'requests >= 2.24.0',
    'pandas >= 1.2.2' 
]

LONG_DESCRIPTION= '''This package provides a Pythonic interface for working with Apache SystemDS.

Apache SystemDS is an open source ML system for the end-to-end data science lifecycle from data integration,
cleaning, and feature engineering, over efficient, local and distributed ML model training,
to deployment and serving.
To facilitate this, bindings from different languages and different system abstractions provide help for:

1. The different tasks of the data-science lifecycle, and
2. users with different expertise.

These high-level scripts are compiled into hybrid execution plans of local, in-memory CPU and GPU operations,
as well as distributed operations on Apache Spark. In contrast to existing systems - that either
provide homogeneous tensors or 2D Datasets - and in order to serve the entire
data science lifecycle, the underlying data model are DataTensors, i.e.,
tensors (multi-dimensional arrays) whose first dimension may have a heterogeneous and nested schema.'''

setup(
    name=ARTIFACT_NAME,
    version=ARTIFACT_VERSION,
    description='Apache SystemDS - An open source ML system for the end-to-end data science lifecycle',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    url='https://github.com/apache/systemds',
    author='SystemDS',
    author_email='dev@systemds.apache.org',
    # Only include the systemds resources not generator and tests.
    packages=find_packages(include=["systemds"]),
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
    python_requires='>=3.6',
    platforms=['Microsoft :: Windows', 'POSIX', 'Unix', 'MacOS'],
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Libraries',
    ],
    license='Apache 2.0',
)
