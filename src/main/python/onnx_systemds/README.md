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
 python3 -m onnx_systemds.convert onnx_systemds/tests/test_models/simple_mat_add.onnx
```



### Run Tests

Form within the `onnx_systemds/tests` directory call:

```bash
export PYTHONPATH="../.."
python3 test_simple.py
```

