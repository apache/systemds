# Performance Testing Algorithms User Manual

This user manual contains details on how to conduct automated performance tests. Work was mostly done in this [PR](https://github.com/apache/systemml/pull/537) and part of [SYSTEMML-1451](https://issues.apache.org/jira/browse/SYSTEMML-1451). Our aim was to move from existing `bash` based performance tests to automatic `python` based automatic performance tests.

### Architecture
Our performance tests suit contains `7` families namely `binomial`, `multinomial`, `stats1`, `stats2`, `regression1`, `regression2`, `clustering`. Within these families we have algorithms grouped under it. Typically a family is a set of algorithms that require the same data generation script. 

- Exceptions: `regression1`, `regression2` and `binomial`. We decide to include these algorithms in separate families to keep the architecture simple.

![System ML Architecture](img/performance-test/perf_test_arch.png)

On a very high level use construct a string with arguments required to run each operation. Once this string is constructed we use the subprocess module to execute this string and extract time from the standard out. 

We also use `json` module write our configurations to a json file. This ensure that our current operation is easy to debug.


We have `5` files in performance test suit `run_perftest.py`, `datagen.py`, `train.py`, `predict.py` and `utils.py`. 

`datagen.py`, `train.py` and `predict.py` generate a dictionary. Our key is the name of algorithm being processed and values is a list with path(s) where all the data required is present. We define this dictionary as a configuration packet.

We will describe each of them in detail the following sections below.

`run_perftest.py` at a high level creates `algos_to_run` list. This list is tuple with key as algorithm and value as family to be executed in our performance test.

In `datagen.py` script we have all functions required to generate data. We return the required configuration packet as a result of this script, that contains key as the `data-gen` script to run and values with location to read data-gen json files from.

In `train.py` script we have functions required to generate training output. We return the required configuration packet as a result of this script, that contains key as the algorithm to run and values with location to read training json files from.

The file `predict.py` contains all functions for all algorithms in the performance test that contain predict script. We return the required configuration packet as a result of this script, that contains key as the algorithm to run and values with location to read predict json files from.

In the file `utils.py` we have all the helper functions required in our performance test. These functions do operations like write `json` files, extract time from std out etc.
 
### Adding New Algorithms
While adding a new algorithm we need know if it has to be part of the any pre existing family. If this algorithm depends on a new data generation script we would need to create a new family. Steps below to take below to add a new algorithm.

Following changes to `run_perftest.py`:

- Add the algorithm to `ML_ALGO` dictionary with its respective family.
- Add the name of the data generation script in `ML_GENDATA` dictionary if it does not exist already.
- Add the name of the training script in `ML_TRAIN` dictionary.
- Add the name of the prediction script in `ML_PREDICT` incase the prediction script exists.

Following changes to `datagen.py`:

- Check if the data generation algorithm has the ability to generate dense and sparse data. If it had the ability to generate only dense data add the corresponding family to `FAMILY_NO_MATRIX_TYPE` list.
- Create a function with `familyname + _ + datagen` with same input arguments namely `matrix_dim`, `matrix_type`, `datagen_dir`.
- Constants and arguments for the data generation script should be defined in function.
- Test the perf test with the algorithm with `mode` as `data-gen`.
- Check output folders, json files, output log.
- Check for possible errors if these folders/files do not exist. (See the troubleshooting section).

Following changes to `train.py`:

- Create the function with `familyname + _ + algoname + _ + train`.
- This function needs to have the following arguments `save_folder_name`, `datagen_dir`, `train_dir`.
- Constants and arguments for the training script should be defined in function.
- Make sure that the return type is a list.
- Test the perf test with the algorithm with `mode` as `train`.
- Check output folders, json files, output log.
- Check for possible errors if these folders/files do not exist. (See the troubleshooting section).

Following changes to `predict.py`:

- Create the function with `algoname + _ + predict`.
- This function needs to have the following arguments `save_file_name`, `datagen_dir`, `train_dir`, `predict_dir`.
- Constants and arguments for the training script should be defined in function.
- Test the perf test with the algorithm with `mode` as `predict`.
- Check output folders, json files, output log.
- Check for possible errors if these folders/files do not exist. (Please see the troubleshooting section).
- Note: `predict.py` will not be executed if the current algorithm being executed does not have predict script.

### Current Default Settings
Default setting for our performance test below:

- Matrix size to 10,000 rows and 100 columns.
- Execution mode `singlenode`.
- Operation modes `data-gen`, `train` and `predict` in sequence.
- Matrix type set to `all`. Which will generate `dense` or / and `sparse` matrices for all relevant algorithms.

### Examples
Some examples of SystemML performance test with arguments shown below:

`./scripts/perftest/python/run_perftest.py --family binomial clustering multinomial regression1 regression2 stats1 stats2
`
Test all algorithms with default parameters.

`./scripts/perftest/python/run_perftest.py --exec-type hybrid_spark --family binomial clustering multinomial regression1 regression2 stats1 stats2
`
Test all algorithms in hybrid spark execution mode.

`./scripts/perftest/python/run_perftest.py --exec-type hybrid_spark --family clustering --mat-shape 10k_5 10k_10 10k_50
`
Test all algorithms in `clustering` family in hybrid spark execution mode, on different matrix size `10k_10` (10,000 rows and 5 columns), `10k_10` and `10k_50`.

`./scripts/perftest/python/run_perftest.py --algo Univar-Stats bivar-stats
`
Run performance test for following algorithms `Univar-Stats` and `bivar-stats`.

`./scripts/perftest/python/run_perftest.py --algo m-svm --family multinomial binomial --mode data-gen train
`
Run performance test for the algorithms `m-svm` with `multinomial` family. Run only data generation and training operations.

`./scripts/perftest/python/run_perftest.py --family regression2 --filename new_log
`
Run performance test for all algorithms under the family `regression2` and log with filename `new_log`.

### Operational Notes
All performance test depend mainly on two scripts for execution `systemml-standalone.py` and `systemml-spark-submit.py`. Incase we need to change standalone or spark parameters we need to manually change these parameters in their respective scripts.

Constants like `DATA_FORMAT` currently set to `csv` and `MATRIX_TYPE_DICT` with `density` set to `0.9` and `sparsity` set to `0.01` are hardcoded in the performance test scripts. They can be changed easily as they are defined at the top of their respective operational scripts.

The logs contain the following information below comma separated.

algorithm | run_type | intercept | matrix_type | data_shape | time_sec
--- | --- | --- | --- | --- | --- | 
multinomial|data-gen|0|dense|10k_100| 0.33
MultiLogReg|train|0|10k_100|dense|6.956
MultiLogReg|predict|0|10k_100|dense|4.780

These logs can be found in `temp` folder (`$SYSTEMML_HOME/scripts/perftest/temp`) in-case not overridden by `--temp-dir`. This `temp` folders also contain the data generated during our performance test.

Every time a script executes in `data-gen` mode successfully, we write a `_SUCCESS` file. If this file exists we ensures that re-run of the same script is not possible as data already exists.

### Troubleshooting
We can debug the performance test by making changes in the following locations based on 

- Please see `utils.py` function `exec_dml_and_parse_time`. In  uncommenting the debug print statement in the function `exec_dml_and_parse_time`. This allows us to inspect the subprocess string being executed.
- Please see `run_perftest.py`. Changing the verbosity level to `0` allows us to log more information while the script runs.
- Eyeballing the json files generated and making sure the arguments are correct.
