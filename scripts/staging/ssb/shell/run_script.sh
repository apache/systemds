
#!/bin/bash
#Mark as executable.
#chmod +x run_script.sh

# Read the database credentials from .env file.
source $PWD/.env

# Variables and arguments.
PG_CONTAINER="ssb-postgres-1"

#https://stackoverflow.com/questions/7069682/how-to-get-arguments-with-flags-in-bash

#Initial variable values.
QUERY_NAME="q2_1"
SCALE=0.1
DB_SYSTEM="systemds"

isQflag=0
isSflag=0
isDflag=0
# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Test environment with SSB Data Loader ===${NC}\n"

#https://unix.stackexchange.com/questions/129391/passing-named-arguments-to-shell-scripts
# Parsing the argument flags.
while getopts "q:s:d:h:" opt; do
    case ${opt} in
        q)  QUERY_NAME="$OPTARG"
            isQflag=1;;
        s)  SCALE=$OPTARG
            isSflag=1;;
        d)  DB_SYSTEM="$OPTARG"
            isDflag=1;;
        h)  echo "Help:"
            cat < other/script_flags_help.txt;;
        \?) echo "Option ${opt} not found. Try again."
            echo "Please use: $0 -q [YOUR_QUERY_NAME] -s [SCALE] -d [DB_SYSTEM]";;
    esac
    case $OPTARG in
        -*) echo "Option ${opt} should have an argument.";;
    esac
done
echo "isQflag=$isQflag"
echo "isSflag=$isSflag"
echo "isDflag=$isDflag"
if [ isQflag==0 ]; then
    echo "Warning: q-flag [QUERY_NAME] is empty. The default q is q2_1."
fi
if [ isSflag==0 ]; then
    echo "Warning: s-flag [SCALE] is empty. The default s is 0.01."
fi
if [ isDflag==0 ]; then
    echo "Warning: d-flag [DATABASE] is empty. The default d is systemds."
fi

echo "Arg 0 (SHELL_SCRIPT): $0"
echo "Arg 1 (QUERY_NAME): ${QUERY_NAME}"
echo "Arg 2 (SCALE): ${SCALE}"
echo "Arg 3 (DB_SYSTEM): ${DB_SYSTEM}"
exit
# Check for the existing required packages. If not install them.
isAllowed="no"
echo -e "${GREEN}Install required packages${NC}"
echo -e "${GREEN}Check whether the following packages exist:${NC}"
echo "docker 'docker compose' git gcc cmake make"

#.
for package in docker git gcc cmake make; do
    if [ ! "$(${package} --version)" ]; then
        echo -e "${BLUE} ${package} package is required for this test bench. Do you want to allow the installation? (yes/no)${NC}"
        read -r isAllowed
        while [ "${isAllowed}" != "yes" ] || [ "${isAllowed}" != "y" ]; do
            echo "Your anwser is ${isAllowed}."
            if [ "${isAllowed}" == "yes" ] || [ "${isAllowed}" == "y" ]; then
                echo "sudo apt-get install ${package}."
                sudo apt-get install ${package}
            elif [ "${isAllowed}" == "no" ] || [ "${isAllowed}" == "n" ]; then
                echo -e "${RED}Sorry, we cannot continue with that test bench without the required packages. The test bench is stopped.${NC}"
                exit
            else
                echo "Your answer '${isAllowed}' is neither 'yes' or 'no'. Please try again."
                read -r isAllowed
            fi
            
        done
    fi
done

if [ ! "$(docker compose version)" ]; then
    echo -e "${BLUE}docker compose is required for this test bench. Do you want to allow the installation? (yes/no)${NC}"
    read -r isAllowed
    while [ "${isAllowed}" != "yes" ] || [ "${isAllowed}" != "y" ]; do
        if [ ${isAllowed} == "yes" ]; then
            echo "sudo apt-get install docker-compose-plugin"
            sudo apt-get install docker-compose-plugin
        elif [ "${isAllowed}" == "no" ] || [ "${isAllowed}" == "n" ]; then
            echo -e "${RED}Sorry, we cannot continue with that test bench without the required packages. The test bench is stopped.${NC}"
            exit
        else
            echo "Your answer '${isAllowed}' is neither 'yes' or 'no'. Please try again."
        fi
        read -r isAllowed
    done
fi

# Check whether the data directory exists.
echo -e "${GREEN}Check for existing data directory and prepare the ssb-dbgen${NC}"
if [ ! -d ssb-dbgen ]; then
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
mkdir -p ../data_dir
mv *.tbl ../data_dir

# Go back to ssb home directory
cd ..
if [ "${DB_SYSTEM}" == "systemds" ] || [ "${DB_SYSTEM}" == "all" ] ; then
    docker pull apache/systemds:latest
    echo -e "${GREEN}Execute DML queries in SystemDS${NC}"
    QUERY_NAME=$(echo "${QUERY_NAME}" | sed 's/\./_/')
    
    ##all: {"q1_1","q1_2","q1_3","q2_1","q2_2","q2_3","q3_1","q3_2","q3_3","q3_4","q4_1","q4_2","q4_3"} 
    if [ "${QUERY_NAME}" == "all" ]; then
        echo "Execute all 13 queries."
        for q in {"q1_1","q1_2","q1_3","q2_1","q2_2","q2_3","q3_1","q3_2","q3_3","q3_4","q4_1","q4_2","q4_3"}
        do  
            echo "Execute query ${QUERY_NAME}.dml"
            docker run -it --rm -v $PWD:/scripts/ apache/systemds:latest -f /scripts/queries/${q}.dml -nvargs input_dir="/scripts/data_dir"
        done
    else
        echo "Execute query ${QUERY_NAME}.dml"
        docker run -it --rm -v $PWD:/scripts/ apache/systemds:latest -f /scripts/queries/${QUERY_NAME}.dml -nvargs input_dir="/scripts/data_dir"
    fi
fi

if [ "${DB_SYSTEM}" == "postgres" ] || [ "${DB_SYSTEM}" == "all" ] ; then
    #Look more in the documentation.
    #https://docs.docker.com/reference/cli/docker/container/ls/
    if [ "$(docker ps -a --filter name=${PG_CONTAINER})" ]; then
        if [ ! "$(docker ps --filter name=${PG_CONTAINER})" ]; then
            echo "Starting existing container..."
            docker start ${PG_CONTAINER}
        fi
    else
        echo "Creating new PostgreSQL container..."
        docker compose up --build -d
    fi
    # Load data and copy into the database
    docker cp data_dir ${PG_CONTAINER}:/tmp
    for table in customer part supplier date lineorder; do
        #docker exec -i ${PG_CONTAINER} ls
        docker exec -i ${PG_CONTAINER} sed -i 's/|$//' "${table}.tbl"
        docker exec -i ${PG_CONTAINER} psql -U ${POSTGRES_USER} -d ${POSTGRES_DB} -c "TRUNCATE TABLE ${table} CASCADE; COPY ${table} FROM '/tmp/${table}.tbl' DELIMITER '|';"
    done
    # Change query_name e.g. from q1_1 to q1.1
    QUERY_NAME=$(echo "${QUERY_NAME}" | sed 's/_/./')
    echo -e "${GREEN}Execute SQL queries in PostgresSQL${NC}"

    ##all: {"q1_1","q1_2","q1_3","q2_1","q2_2","q2_3","q3_1","q3_2","q3_3","q3_4","q4_1","q4_2","q4_3"} 
    if [ "${QUERY_NAME}" = "all" ]; then
        echo "Execute all 13 queries."
        for q in {"q1.1","q1.2","q1.3","q2.1","q2.2","q2.3","q3.1","q3.2","q3.3","q3.4","q4.1","q4.2","q4.3"}; do  
            echo "Execute query ${q}.sql"
            docker exec -i ${PG_CONTAINER} psql -U ${POSTGRES_USER} -d ${POSTGRES_DB}  < sql/${q}.sql
       done
    else
        echo "Execute query ${QUERY_NAME}.sql"
        echo "docker exec -i ${PG_CONTAINER} psql -U ${POSTGRES_USER} -d ${POSTGRES_DB}  < sql/${QUERY_NAME}.sql"
        docker exec -i ${PG_CONTAINER} psql -U ${POSTGRES_USER} -d ${POSTGRES_DB}  < sql/${QUERY_NAME}.sql
    fi
fi

#TODO Add duckdb support
#if [ "${DB_SYSTEM}" == "duckdb" ] || [ "${DB_SYSTEM}" == "all" ]; then
#    # Change query_name e.g. from q1_1 to q1.1
#    QUERY_NAME=$(echo "${QUERY_NAME}" | sed 's/_/./')
#    echo -e "${GREEN}Execute SQL queries in DuckDB${NC}"

    ##all: {"q1_1","q1_2","q1_3","q2_1","q2_2","q2_3","q3_1","q3_2","q3_3","q3_4","q4_1","q4_2","q4_3"} 
#    if [ "${QUERY_NAME}" = "all" ]; then
#        echo "Execute all 13 queries."
#        for q in {"q1.1","q1.2","q1.3","q2.1","q2.2","q2.3","q3.1","q3.2","q3.3","q3.4","q4.1","q4.2","q4.3"}
#        do  
#            echo "Execute query ${QUERY_NAME}."
#            #TODO
#       done
#    else
#        echo "Execute query ${QUERY_NAME}"
#        #TODO
#    fi
#fi