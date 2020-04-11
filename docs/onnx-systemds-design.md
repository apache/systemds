# onnx-systemds

A tool for importing [ONNX](https://github.com/onnx/onnx/blob/master/docs/IR.md) graphs into SystemDS



## Goals

* Initially support for [operators of the ONNX base definition](https://github.com/onnx/onnx/blob/master/docs/Operators.md)

* additionally support for [operators defined by ONNX-ML](https://github.com/onnx/onnx/blob/master/docs/Operators-ml.md)

  

## Limitations

* not able to support all data types / operators as they are not currently supported by SystemDS



## Suggested Implementation

Since the ONNX specification includes the conditional operators [loop](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Loop) and [if](https://github.com/onnx/onnx/blob/master/docs/Operators.md#If) a direct conversion from ONNX to the internal HOP might not be ideal. 

Hence my suggested implementation is a dedicated tool invoked from command line which generates dml scripts. This also enables optimizations performed by the compiler. 

### Example Call

```bash
onnx-systemds model.onx --out model_script.dml
```


### Tooling

* Due to the availability of a  [Python API](https://github.com/onnx/onnx/blob/master/docs/PythonAPIOverview.md) for ONNX I would suggest implementing the tool in Python
* Another advantage of Python is good support for template engines e.g. [Jinja](https://jinja.palletsprojects.com/en/2.11.x/)
  * An implementation could use templates for various operators which are then combined into a script

### Implementation Details

ONNX is a [serialized graph](https://github.com/onnx/onnx/blob/master/docs/IR.md#graphs) structured as a sorted list of nodes that form a graph which is free of cycles

1. Loading in the serialized structure
2. [Checking](https://github.com/onnx/onnx/blob/master/docs/PythonAPIOverview.md#checking-an-onnx-model) model and [converting](https://github.com/onnx/onnx/blob/master/docs/PythonAPIOverview.md#converting-version-of-an-onnx-model-within-default-domain-aionnx) models to a common version
3. Building an internal graph-structure
4. Generating the dml script while traversing this graph
   * provided information in doc_strings and other description variables are added as comments to improve human-readability of the generated script

