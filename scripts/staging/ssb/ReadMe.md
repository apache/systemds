# Star Schema Benchmark (SSB) for SystemDS [SystemDS-3862](https://issues.apache.org/jira/browse/SYSTEMDS-3862) 


## Foundation:
- There are [13 queries already written in SQL](https://github.com/apache/doris/tree/master/tools/ssb-tools/ssb-queries).
- There are existing DML relational algebra operators raSelect(), raJoin() and raGroupBy().
- Our task is to implement the DML version of these queries to run them in SystemDS.
- There are existing DML query implementations ([Git request](https://github.com/apache/systemds/pull/2280) and [code](https://github.com/apache/systemds/tree/main/scripts/staging/ssb)) of the previous group which are a bit slow and contain errors. They also provided longer scripts to run experiments in SystemDS, Postgres and DuckDB.

## Setup
1. First, install [Docker](https://docs.docker.com/get-started/get-docker/), [Docker Compose](https://docs.docker.com/compose/install/) and its necessary libraries.
  
  For Ubuntu, there is the following tutorials [for Docker](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository) and [Docker Compose](https://docs.docker.com/compose/install/linux/#install-using-the-repository) using apt repository. You can add [Docker Desktop](https://docs.docker.com/desktop/setup/install/linux/ubuntu/), too.
  

1. Now, follow the tutorials to install Docker versions of database systems [SystemDS](https://apache.github.io/systemds/site/docker)
  

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

## Run scripts
### Using shell script
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
### Using docker compose
To run our queries, we can use docker compose (in ssb directory). We can run in one mode.
- A selected query

```
docker-compose up --build
docker-compose up
```
Create an .env file and modify before each "docker compose up".
```
# in .env file
SCALE=[OUR_VALUE]
QUERY=[OUR_QUERY_NUMBER]
```
```
#Example:
# in .env file
SCALE=0.01
QUERY=q1_1
```
### Further considerations.
To compare the correctness and do benchmarks, PostgreSQL can be used.