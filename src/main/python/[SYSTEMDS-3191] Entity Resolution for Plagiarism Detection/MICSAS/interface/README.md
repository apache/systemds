# MISIM Interface

## Setup

Use this [Makefile](../Makefile).

Vocabularies are available [here](https://www.dropbox.com/s/zilq32a4s9pygde/datasets.tar.xz).
Pre-trained models are available [here](https://www.dropbox.com/s/jlfp2oypzkc29q7/models.tar.xz).
Extract them into `../data/`.

## Usage example:
```python
import misim.interface as misim

cass_manager = misim.CASSManager()
gnn_preprocessor = misim.GNNPreprocessor('misim/data/datasets/poj/dataset-gnn/vocab.pkl')
gnn_runner = misim.GNNRunner('misim/data/datasets/poj/dataset-gnn/vocab.pkl', 'misim/data/models/poj/gnn/0/model.pt')

# Compute GNN feature vectors for each function/loop in a source file.
cass_strs = cass_manager.extract_cass_strs_from_src_file('test.c', extract_loops=True)
casses, src_ranges = cass_manager.load_casses_from_strs(cass_strs)
inputs = gnn_preprocessor.preprocess_casses_seperated(casses)
vectors = gnn_runner.compute_code_vector_batched(inputs)
for i in range(len(src_ranges)):
    print(src_ranges[i], vectors[i])

# Compute code similarity between two source files.
cass_strs_1 = cass_manager.extract_cass_strs_from_src_file('test1.c', extract_loops=False)
cass_strs_2 = cass_manager.extract_cass_strs_from_src_file('test2.c', extract_loops=False)
casses_1, _ = cass_manager.load_casses_from_strs(cass_strs_1)
casses_2, _ = cass_manager.load_casses_from_strs(cass_strs_2)
input_1 = gnn_preprocessor.preprocess_casses_combined(casses_1)
input_2 = gnn_preprocessor.preprocess_casses_combined(casses_2)
vectors = gnn_runner.compute_code_vector_batched([input1, input2])
from numpy.linalg import norm
similarity = (vector[0] @ vector[1].T) / (norm(vector[0]) * norm(vector[1]))
```
