# Star Schema Benchmark (SSB) for SystemDS [SystemDS-3862](https://issues.apache.org/jira/browse/SYSTEMDS-3862) 


## Foundation:
- There are [13 queries already written in SQL](https://github.com/apache/doris/tree/master/tools/ssb-tools/ssb-queries).
- There are existing DML relational algebra operators raSelect(), raJoin() and raGroupBy().
- Our task is to implement the DML version of these queries to run them in SystemDS.
- There are existing DML query implementations ([Git request](https://github.com/apache/systemds/pull/2280) and [code](https://github.com/apache/systemds/tree/main/scripts/staging/ssb)) of the previous group which are a bit slow and contain errors.

## General steps
- Prepare the setup.
- Translate/rewrite the queries into DML language to run them in SystemDS.
- Therefore, we should use these relational algebra operators in DML.
- Use [SSB generator](https://github.com/eyalroz/ssb-dbgen) to generate data.
- Run ssh scripts for experiments in the selected database systems. Use also scale factors.
- Compare the runtime of each query in each system. 
## Run 
To run our queries, we can execute the following **run_script.sh** script (in ssb directory). We can run in both modes.
- All queries
- A selected query
```
./run_script.sh all [SCALE] # For all queries
./run_script.sh [QUERY_NUMBER] [SCALE] # For a selected query
```
Example
```
./run_script.sh all 0.1
./run_script.sh q_4_3 0.1
```
### Further expansion:
Using docker compose.

## Setup
(To run without the shell script)
1. First, install [Docker](https://docs.docker.com/get-started/get-docker/), [Docker Compose](https://docs.docker.com/compose/install/) and its necessary libraries.
  
  For Ubuntu, there is the following tutorials [for Docker](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository) and [Docker Compose](https://docs.docker.com/compose/install/linux/#install-using-the-repository) using apt repository. You can add [Docker Desktop](https://docs.docker.com/desktop/setup/install/linux/ubuntu/), too.
  

1. Now, follow the tutorials to install Docker versions of database systems [SystemDS](https://apache.github.io/systemds/site/docker), [PostgreSQL](https://hub.docker.com/_/postgres), ....
  

If the example in the SystemDS link does not work, use that code line instead. Create a DML file, open its directory and execute the code.
```
docker run -it --rm -v $PWD:/scripts apache/systemds:latest -f /scripts/[file_name].dml
# Example
docker run -it --rm -v $PWD:/scripts apache/systemds:latest -f /scripts/hello.dml
```
3. Clone the git repository of [ssb-dbgen (SSB data set generator)](https://github.com/eyalroz/ssb-dbgen/tree/master) and generate data with it. 
```
# Build the generator
cmake -B ./build && cmake --build ./build
# Run the generator (with -s )
build/dbgen -b dists.dss -v -s 1
```
For more options look into the original documentation. 

Run with:
```
docker run -it --rm -v $PWD:/scripts/ apache/systemds:latest -f /scripts/queries/[QUERY_NUMBER].dml -nvargs input_dir="/scripts/data/..."
docker run -it --rm -v $PWD:/scripts/ apache/systemds:latest -f /scripts/queries/q4_3.dml -nvargs input_dir="/scripts/data/..."
```

To compare the correctness and do benchmarks, PostgreSQL can be used.