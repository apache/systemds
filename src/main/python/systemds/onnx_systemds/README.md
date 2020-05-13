# onnx-systemds

A tool for importing/exporting [ONNX](https://github.com/onnx/onnx/blob/master/docs/IR.md) graphs into/from SystemDS DML scripts.

For a more detailed description of this converter refer to the [description of the converter design](docs/onnx-systemds-design.md)

## Prerequisites

to run onnx-systemds you need [onnx](https://github.com/onnx/onnx)

* [Installation instructions](https://github.com/onnx/onnx#installation)
* The conda install seems to work best 
* the environment variable `SYSTEMDS_ROOT` needs to be set to the root of the repository

## Usage

An example call from the `src/main/python` directory of systemds:

```bash
python -m systemds.onnx_systemds.convert tests/onnx/test_models/simple_mat_add.onnx
```

This will generate the dml script `simple_mat_add.dml` in your current directory. 

### Run Tests

Form the `src/main/python` directory of systemds:

At first generate the test models:

```bash
python tests/onnx/test_models/model_generate.py
```

Then you can run the tests:

```bash
python -m unittest tests/onnx/test_simple.py
```
