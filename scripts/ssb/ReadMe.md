# Star Schema Benchmark (SSB) for SystemDS [SystemDS-3862](https://issues.apache.org/jira/browse/SYSTEMDS-3862) 


## Foundation:
- There are [13 queries already written in SQL](https://github.com/apache/doris/tree/master/tools/ssb-tools/ssb-queries).
- There are existing DML relational algebra operators raSelect(), raJoin() and raGroupBy().
- Our task is to implement the DML version of these queries to run them in SystemDS.
- There are existing DML query implementations ([Git request](https://github.com/apache/systemds/pull/2280) and [code](https://github.com/ghafek/systemds/tree/feature/ssb-benchmark/scripts/ssb)) of the previous group which are a bit slow and contain errors.
   
## Setup

- First, install [Docker](https://docs.docker.com/get-started/get-docker/) and its necessary libraries.
  
  For Ubuntu, there is the [following tutorial using apt repository](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository). You can add [Docker Desktop](https://docs.docker.com/desktop/setup/install/linux/ubuntu/), too.
  

- Now, follow the tutorials to install Docker versions of database systems [SystemDS](https://apache.github.io/systemds/site/docker), [PostgreSQL](https://hub.docker.com/_/postgres), ....
  

If the example in the SystemDS link does not work, use that code line instead. Create a DML file, open its directory and execute the code.
```
docker run -it --rm -v $PWD:/scripts apache/systemds -f /scripts/[file_name].dml
# Example
docker run -it --rm -v $PWD:/scripts apache/systemds -f /scripts/hello.dml
```
--- SSB ... 

## General steps
- Prepare the setup.
- Translate/rewrite the queries into DML language to run them in SystemDS.
- Therefore, we should use these relational algebra operators in DML.
- Use [SSB generator](https://github.com/eyalroz/ssb-dbgen) to generate data.
- Run ssh scripts for experiments in the selected database systems. Use also scale factors.
- Compare the runtime of each query in each system. 
