.. ------------------------------------------------------------------------------
..  Copyright 2020 Graz University of Technology
..
..  Licensed under the Apache License, Version 2.0 (the "License");
..  you may not use this file except in compliance with the License.
..  You may obtain a copy of the License at
..
..    http://www.apache.org/licenses/LICENSE-2.0
..
..  Unless required by applicable law or agreed to in writing, software
..  distributed under the License is distributed on an "AS IS" BASIS,
..  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
..  See the License for the specific language governing permissions and
..  limitations under the License.
.. ------------------------------------------------------------------------------

SystemDS
========

This package provides a Pythonic interface for working with SystemDS.

SystemDS is a versatile system for the end-to-end data science lifecycle from data integration,
cleaning, and feature engineering, over efficient, local and distributed ML model training,
to deployment and serving.
To facilitate this, bindings from different languages and different system abstractions provide help for:

1. The different tasks of the data-science lifecycle, and
2. users with different expertise.

These high-level scripts are compiled into hybrid execution plans of local, in-memory CPU and GPU operations,
as well as distributed operations on Apache Spark. In contrast to existing systems - that either
provide homogeneous tensors or 2D Datasets - and in order to serve the entire
data science lifecycle, the underlying data model are DataTensors, i.e.,
tensors (multi-dimensional arrays) whose first dimension may have a heterogeneous and nested schema.

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting Started:

   install.rst
   simple_examples.rst

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Central Classes

   matrix.rst
