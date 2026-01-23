
#!/bin/bash
#Mark as executable.
#chmod +x run_script.sh

QUERY_NAME=$1
SCALE=$2
DB_SYSTEM=$3

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Test environment with SSB Data Loader ===${NC}\n"

echo "Arg 0 (SHELL_SCRIPT): $0"
echo "Arg 1 (QUERY_NAME): ${QUERY_NAME}"
echo "Arg 2 (SCALE): ${SCALE}"
echo "Arg 3 (DB_SYSTEM): ${DB_SYSTEM}"
echo "Arg 3 (DB_SYSTEM): ${DB_SYSTEM}"
# Install docker.
echo -e "${GREEN}Install packages${NC}"
echo -e "${BLUE}sudo apt install docker git gcc cmake make${NC}"
sudo apt install docker git gcc cmake make

# Check whether the data directory exists.
cd ..
echo -e "${GREEN}Check for existing data directory and prepare the ssb-dbgen${NC}"
if [ ! -d ssb-dbgen ]; then
    mkdir data_dir
    git clone https://github.com/eyalroz/ssb-dbgen.git --depth 1
    cd ssb-dbgen
else
    cd ssb-dbgen
    git pull
fi

echo -e "${GREEN}Build ssb-dbgen and generate data with a given scale factor${NC}"
# Build the generator
cmake -B ./build && cmake --build ./build
# Run the generator (with -s )
build/dbgen -b dists.dss -v -s $SCALE
mv *.tbl ../data_dir
$DB_SYSTEM == "dbsystemds" || 

if [ ${DB_SYSTEM} == "systemds" ] || [ ${DB_SYSTEM} == "all" ] ; then
    docker pull apache/systemds:latest
    echo -e "${GREEN}Execute DML queries in SystemDS${NC}"
    
    ##all: {"q1_1","q1_2","q1_3","q2_1","q2_2","q2_3","q3_1","q3_2","q3_3","q3_4","q4_1","q4_2","q4_3"} 
    if [ $QUERY_NAME == "all" ]; then
        echo "Execute all 13 queries."
        for q in {"q1_1","q1_2","q1_3","q2_1","q2_2","q2_3","q3_1","q3_2","q3_3","q3_4","q4_1","q4_2","q4_3"}
        do  
            echo "Execute query ${QUERY_NAME}."
            docker run -it --rm -v $PWD:/scripts/ apache/systemds:latest -f /scripts/queries/${q}.dml -nvargs input_dir="/scripts/data_dir"
        done
    else
        echo "Execute query ${QUERY_NAME}"
        docker run -it --rm -v $PWD:/scripts/ apache/systemds:latest -f /scripts/queries/${QUERY}.dml -nvargs input_dir="/scripts/data_dir"
    fi
fi

if [ ${DB_SYSTEM} == "postgres" ] || [ ${DB_SYSTEM} == "all" ] ; then
    #TODO Maybe check if the db exists and just do docker cp src_dir dest_dir and SQL COPY?
    docker compose up --build
    # Change query_name e.g. from q1_1 to q1.1
    QUERY_NAME=$(echo "${QUERY_NAME}" | sed 's/_/./')
    echo -e "${GREEN}Execute SQL queries in PostgresSQL${NC}"

    ##all: {"q1_1","q1_2","q1_3","q2_1","q2_2","q2_3","q3_1","q3_2","q3_3","q3_4","q4_1","q4_2","q4_3"} 
    if [ $QUERY_NAME = "all" ]; then
        echo "Execute all 13 queries."
        for q in {"q1.1","q1.2","q1.3","q2.1","q2.2","q2.3","q3.1","q3.2","q3.3","q3.4","q4.1","q4.2","q4.3"}
        do  
            echo "Execute query ${QUERY_NAME}."
            docker exec -i postgres-ssb psql -U ${POSTGRES_USER} -d ${POSTGRES_DB}  < sql/${QUERY_NAME}.sql
       done
    else
        echo "Execute query ${QUERY_NAME}"
        docker exec -i postgres-ssb psql -U ${POSTGRES_USER} -d ${POSTGRES_DB}  < sql/${QUERY_NAME}.sql
    fi
fi

#TODO Add duckdb support
if [ ${DB_SYSTEM} == "duckdb" ] || [ ${DB_SYSTEM} == "all" ] ; then
    # Change query_name e.g. from q1_1 to q1.1
    QUERY_NAME=$(echo "${QUERY_NAME}" | sed 's/_/./')
    echo -e "${GREEN}Execute SQL queries in DuckDB${NC}"

    ##all: {"q1_1","q1_2","q1_3","q2_1","q2_2","q2_3","q3_1","q3_2","q3_3","q3_4","q4_1","q4_2","q4_3"} 
    if [ $QUERY_NAME = "all" ]; then
        echo "Execute all 13 queries."
        for q in {"q1.1","q1.2","q1.3","q2.1","q2.2","q2.3","q3.1","q3.2","q3.3","q3.4","q4.1","q4.2","q4.3"}
        do  
            echo "Execute query ${QUERY_NAME}."
            #TODO
       done
    else
        echo "Execute query ${QUERY_NAME}"
        #TODO
fi