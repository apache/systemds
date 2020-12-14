# Sklearn - Importer
[Sklearn](https://scikit-learn.org/stable/index.html) is a very popular and well established open-source python library for data science applications. A large number of common algorithms and many useful tools are implemented and maintained. 

### Idea
Allowing the import of sklearn models, allows for an easy extension of already established implementations with systemds.

### Current State
Currently a [ONNX](https://onnx.ai/) importer is in staging [here](/systemds/scripts/staging/onnx), which is somehow broken or rather breaks something else in the stable branch.

This tool allows the conversion of ONNX graphs to dml.

## Sklearn - Importer
We have following idea. Since the ONNX importer is (to some degree) working, we suggest the modification/extension of the importer to work with sklearn models.

For either approach (when reading from a saved model) we expect a [pickle](https://docs.python.org/3/library/pickle.html) serialized python object, since scikit-learn uses Python's built-in persitence model as described [here](https://scikit-learn.org/stable/modules/model_persistence.html). [Joblib](https://joblib.readthedocs.io/en/latest/persistence.html) is a pickle replacement, which works more efficient on large/complex objects, which is the case with some scikit-learn models. In both cases there are some security and maintainability concerns addressed [here](https://scikit-learn.org/stable/modules/model_persistence.html#security-maintainability-limitations).

### Proposal 1
One possible approach to this problem is a direct mapping from scikit-learn to DML. But the effort for this approach may be out of scope for this pull request (for now). 

### Proposal 2
A probably easier approach would involve a indirect mapping to ONNX and then to DML:

sklearn --> onnx && onnx --> dml ==> sklearn --> dml

Sklearn models may be converted to ONNX using the [sklearn-onnx](http://onnx.ai/sklearn-onnx/) converter, part of the official ONNX project. The conversion from ONNX to DML can be accomplished using the existing [ONNX-Importer](/systemds/scripts/staging/onnx) of systemds.

This approach requires fixing the onnx importer and the inclusion of a further dependency (sklearn-onnx). The [package](https://github.com/onnx/sklearn-onnx) is published under a MIT license and requires a few other [dependencies](https://github.com/onnx/sklearn-onnx/blob/master/requirements.txt).

## References:
 * https://scikit-learn.org/stable/index.html
 * https://github.com/onnx/sklearn-onnx
 * http://onnx.ai/sklearn-onnx/
 * https://scikit-learn.org/stable/related_projects.html#related-projects
 * https://scikit-learn.org/stable/modules/model_persistence.html
 

