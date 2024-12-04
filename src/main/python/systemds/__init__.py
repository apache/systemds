# -------------------------------------------------------------
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
# -------------------------------------------------------------
from importlib.metadata import version, PackageNotFoundError
from systemds import context
from systemds import operator
from systemds import examples

__all__ = ["context", "operator", "examples"]

required_packages = [
    ("torch", "2.5.1"),
    ("torchvision", "0.20.1"),
    ("librosa", "0.10.2"),
    ("opencv-python", "4.10.0.84"),
    ("opt-einsum", "3.3.0"),
    ("h5py", "3.11.0"),
    ("transformers", "4.46.3"),
    ("nltk", "3.9.1"),
    ("gensim", "4.3.3"),
]


def check_package_version(package_name, required_version):
    try:
        return version(package_name) >= required_version
    except PackageNotFoundError:
        return False


if all(check_package_version(pkg, version) for pkg, version in required_packages):
    try:
        from systemds import scuro

        __all__.append("scuro")
    except ImportError as e:
        print(f"Scuro could not be imported: {e}")
else:
    missing = [
        f"{pkg} {version}"
        for pkg, version in required_packages
        if not check_package_version(pkg, version)
    ]
    print(
        f"Warning: Scuro dependencies missing or wrong version installed: {', '.join(missing)}"
    )
