SklearnToDMLMapper
==================

SklearnToDMLMapper is a simple tool for transforming scikit-learn pipelines into DML scripts.
This tool may be used over a simple command line interface, where a scikit-learn pipeline provided over a `pickle <https://docs.python.org/3/library/pickle.html>`_ file. Alternatively, SklearnToDMLMapper can be used in a script as a Python module.


Prerequisites
-------------

If a pickle file is provided, no dependecies are necessary except for python 3.6+.
Otherwise, scikit-learn needs to be `installed <https://scikit-learn.org/stable/install.html>`_.

Usage
-----

For usage over the CLI, as example call may look as follows:

    ./SklearnToDMLMapper.py -i input -o output_path pipe.pkl

* input: name (prefix) of the input file(s) (see below)
* output_path: transformed pipeline as .dml script
* pipe.pkl: binary file (pickle) of a sklear pipeline

Used as a Python module a script may look as follows::

    from sklearn.pipeline import make_pipeline
    # Other imports from sklearn
    from SklearnToDMLMapper import SklearnToDMLMapper

    pipeline = make_pipeline(...)

    mapper = SklearnToDMLMapper(pipeline, 'input')
    mapper.transform()
    mapper.save('mapped_pipeline')

or, alternatively using a pickle file::

    from SklearnToDMLMapper import SklearnToDMLMapper

    with open('pipeline.pkl', 'rb') as f:
        pipeline = pickle.load(f)

    mapper = SklearnToDMLMapper(pipeline, 'input')
    mapper.transform()
    mapper.save('mapped_pipeline')

API description
---------------

.. autoclass:: SklearnToDMLMapper