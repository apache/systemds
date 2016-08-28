#!/usr/bin/python
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

from setuptools import setup, find_packages
import os
import time

VERSION = '0.11.0.dev1'
RELEASED_DATE = str(time.strftime("%m/%d/%Y"))
numpy_version = '1.8.2'
scipy_version = '0.15.1'
REQUIRED_PACKAGES = [
    'numpy >= %s' % numpy_version,
    'scipy >= %s' % scipy_version
]

PACKAGE_DATA = []
for path, subdirs, files in os.walk('SystemML/SystemML-java'):
    for name in files:
        PACKAGE_DATA = PACKAGE_DATA + [ os.path.join(path, name).replace('./', '') ]
        
setup(
    name='SystemML',
    version=VERSION,
    description='Apache SystemML is a distributed and declarative machine learning platform.',
    long_description='''
    
    Apache SystemML is an effort undergoing incubation at the Apache Software Foundation (ASF), sponsored by the Apache Incubator PMC.
    While incubation status is not necessarily a reflection of the completeness
    or stability of the code, it does indicate that the project has yet to be
    fully endorsed by the ASF.
    
    Apache SystemML provides declarative large-scale machine learning (ML) that aims at 
    flexible specification of ML algorithms and automatic generation of hybrid runtime 
    plans ranging from single-node, in-memory computations, to distributed computations on Apache Hadoop and Apache Spark.
    
    Note: This is not a released version and was built with SNAPSHOT available on the date''' + RELEASED_DATE,
    url='http://systemml.apache.org/',
    author='Apache SystemML',
    author_email='dev@systemml.incubator.apache.org',
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
    package_data={
        'SystemML-java': PACKAGE_DATA
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