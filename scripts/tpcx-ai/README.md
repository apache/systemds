# Implementation of TPCx-AI on Apache SystemDS

The TPCx-AI is an express benchmark developed by the TPC (Transaction Processing Performance Council) 
specifically tailored for end-to-end machine learning systems. 
For further information, refer to the official documentation provided by the TPC: 
[TPCx-AI documentation](https://www.tpc.org/TPC_Documents_Current_Versions/pdf/TPCX-AI_v1.0.3.1.pdf)

To run the TPCx-AI benchmark on SystemDS:
 * Download the TPCx-AI benchmark kit from [TPC's website](https://www.tpc.org/tpc_documents_current_versions/current_specifications5.asp).
 * Install and build SystemDS.
 * Build the python package and copy the distribution to the TPCx-AI root directory.
 * Copy the files from this directory (`tpcx-ai`) into the TPCx-AI benchmark kit root directory.
 * Set values for scale factor and java paths in `setenv_sds.sh`.
 * Set up TPCx-AI by running `setup_python_sds.sh`.
 * Generate data with `generate_data.sh`.
 * Lastly, execute the benchmark using `TPCx-AI_Benchmarkrun_sds.sh`.

## Detailed Instructions

### Prerequisites 

The following sections describe system prerequisites and steps to prepare and adapt the TPCx-AI benchmark kit to run on SystemDS.

#### Downloading the TPCx-AI Benchmark Kit

Go to [TPC's website](https://www.tpc.org/tpc_documents_current_versions/current_specifications5.asp) and download the TPCx-AI benchmark kit. 
Extract the archived directory, which from now on will be referred to as TPCx-AI root directory.


#### Building SystemDS

Go back to the SystemDS root directory and follow the installation guide for SystemDS: <http://apache.github.io/systemds/site/install>. 
Build the project with maven. 
```bash
mvn package -P distribution
```

#### Building python package and copy to TPCx-AI root directory

- From `SYSTEMDS_ROOT/src/main/python` run `create_python_dist.py`.

```bash
python3 create_python_dist.py
```

- Now, in the `./dist` directory, there will exist the source distribution `systemds-VERSION.tar.gz`
  and the wheel distribution `systemds-VERSION-py3-none-any.whl`, with `VERSION` being the current version number
- copy the `systemds-VERSION-py3-none-any.whl` to the TPCx-AI benchmark kit root directory

#### Transfering Files to TPCx-AI Root Directory

The following files need to be copied from this directory into the TPCx-AI root directory:

- generate_data.sh
- setenv_sds.sh
- setup_python_sds.sh
- TPCx-AI_Benchmarkrun_sds.sh

The following directories in the TPCx-AI benchmark kit directory need to be **replaced**:

- Replace the driver directory with the driver directory from this directory.
- In the TPCx-AI root directory, navigate to workload/python/workload and replace the 10 use case files with the files in the `use_cases` directory.
- Replace `tpcxai_fdr.py` and `tpcxai_fdr_template.html` from the `TPCx-AI_ROOT\tools directory with the files in this directory.

#### Setting Up TPCx-AI 

Now the benchmark kit is ready for set up and installation.
Prerequisites for running are: 
* Java 8, 
* Java 11, 
* Python 3.6+ 
* Anaconda3/Conda4+ 

* The binaries "java", "sbt" and "conda" must be included (and have "priority") in the PATH environment variable.
* Disk Space: Make sure you have enough disk space to store the test data that will be generated, in the `output/raw_data`
The value of "TPCxAI_SCALE_FACTOR" in the setenv.sh file will determine the approximate size (GB) of the dataset that will be generated and used during the benchmark execution.

For more detailed information and optional setup possibilities refer to the official TPCs-AI documentation: 
[TPCx-AI documentation](https://www.tpc.org/TPC_Documents_Current_Versions/pdf/TPCX-AI_v1.0.3.1.pdf).

#### Setting up Environment in setenv_sds.sh

There are three variables that need to be set in the setenv_sds.sh files prior to set-up:
* JAVA8_HOME: Set this variable to the path to the home directory of your Java 8 version.
* JAVA11_HOME: Set this variable to the path to the home directory of your Java 11 version.
* TPCxAI_SCALE_FACTOR: Set to the desired scale factor of the data set; the default is 1.

#### Running the Setup Script

* This implementation is based on SystemdDS Version 3.3.0 (commit id: 5ad67e8). If you want to use a different version, make sure to set the correct filename for the systemds distribution in the `setup_python_sds.sh`file.
The filename should match the appropriate version and build of systemds for your environment.
* Run `setup_python_sds.sh` to automatically set up the benchmark, 
install all the neccessary libraries, install SystemDS as and set up the virtual environments. 

## Benchmark Execution

### Data Generation

Before running the benchmark, data needs to be generated. To generate data, run the `generate_data.sh` script. The size of the generated data can be chosen by setting the 
TPCxAI_SCALE_FACTOR variable from the setenv_sds.sh file. The default value is 1, which leads to the generation of a dataset with the size of 1 GB.

### Running the Benchmark

Now the benchmark can be executed by running the `TPCx-AI_Benchmarkrun_sds.sh` script.

```bash
./TPCx-AI_Benchmarkrun_sds.sh
```