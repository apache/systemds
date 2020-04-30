# onnx-systemds

A tool for importing/exporting [ONNX](https://github.com/onnx/onnx/blob/master/docs/IR.md) graphs into/from SystemDS DML scripts.



## Prerequisites

to run onnx-systemds you need [onnx](https://github.com/onnx/onnx)

* [Installation instructions](https://github.com/onnx/onnx#installation)



## Usage

For running onnx-systemds the environment variable `SYSTEMDS_ROOT` needs to be set. 

An example call from the `src/main/python` directory of systemds:

```bash
 python3 -m onnx_systemds.convert onnx_systemds/tests/test_models/simple_mat_add.onnx
```



### Run Tests

Again `SYSTEMDS_ROOT` needs to be set. 

Form within the `onnx_systemds/tests` directory call:

```bash
export PYTHONPATH="../.."
python3 test_simple.py
```

