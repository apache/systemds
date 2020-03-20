# Python Readme

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
