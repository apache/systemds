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

from __future__ import print_function
import os
import sys
from setuptools import find_packages, setup
import time

try:
    exec(open('systemml/project_info.py').read())
except IOError:
    print("Could not read project_info.py. Will use default values.", file=sys.stderr)
    BUILD_DATE_TIME = str(time.strftime("%Y%m%d.%H%M%S"))
    __project_artifact_id__ = 'systemml'
    __project_version__ = BUILD_DATE_TIME + '.dev0'
ARTIFACT_NAME = __project_artifact_id__
ARTIFACT_VERSION = __project_version__
ARTIFACT_VERSION_SHORT = ARTIFACT_VERSION.split("-")[0]

numpy_version = '1.8.2'
scipy_version = '0.15.1'
pillow_version = '2.0.0'
REQUIRED_PACKAGES = [
    'numpy >= %s' % numpy_version,
    'scipy >= %s' % scipy_version,
    'pandas',
    'Pillow >= %s' % pillow_version
]


python_dir = 'systemml'
java_dir='systemml-java'
java_dir_full_path = os.path.join(python_dir, java_dir)
PACKAGE_DATA = []
for path, subdirs, files in os.walk(java_dir_full_path):
    for name in files:
        PACKAGE_DATA = PACKAGE_DATA + [ os.path.join(path, name).replace('./', '') ]
PACKAGE_DATA = PACKAGE_DATA + [os.path.join(python_dir, 'LICENSE'),
                               os.path.join(python_dir, 'NOTICE')]

setup(
    name=ARTIFACT_NAME,
    version=ARTIFACT_VERSION_SHORT,
    description='Apache SystemML is a distributed and declarative machine learning platform.',
    long_description='''

    Apache SystemML provides declarative large-scale machine learning (ML) that aims at
    flexible specification of ML algorithms and automatic generation of hybrid runtime
    plans ranging from single-node, in-memory computations, to distributed computations on Apache
    Hadoop and Apache Spark.
    ''',
    url='http://systemml.apache.org/',
    author='Apache SystemML',
    author_email='dev@systemml.apache.org',
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
    package_data={
        'systemml-java': PACKAGE_DATA
    },
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Libraries',
        ],
    license='Apache 2.0',
    )
