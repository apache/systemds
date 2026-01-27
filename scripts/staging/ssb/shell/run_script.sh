
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
isGflag=0

dml_query_array=("q1_1" "q1_2" "q1_3" "q2_1" "q2_2" "q2_3" "q3_1" "q3_2" "q3_3" "q3_4" "q4_1" "q4_2" "q4_3")
sql_query_array=("q1.1" "q1.2" "q1.3" "q2.1" "q2.2" "q2.3" "q3.1" "q3.2" "q3.3" "q3.4" "q4.1" "q4.2" "q4.3")

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Test environment for SSB Data ===${NC}\n"

#https://unix.stackexchange.com/questions/129391/passing-named-arguments-to-shell-scripts
# Parsing the argument flags.
while getopts "q:s:d:gh" opt; do
    case ${opt} in
        q)  QUERY_NAME="$OPTARG"
            isQflag=1;;
        s)  SCALE=$OPTARG
            isSflag=1;;
        d)  DB_SYSTEM="$OPTARG"
            isDflag=1;;
        g)  isGflag=1;;
        #h (help) without flags
        h)  echo "Help:" 
            cat < other/script_flags_help.txt
            echo "Thank you.";;
        ?) echo "Option ${opt} not found. Try again."
            echo "Please use: $0 -q [YOUR_QUERY_NAME] -s [YOUR_SCALE] -d [YOUR_DB_SYSTEM]";;
    esac
    case $OPTARG in
        -*) echo "Option ${opt} should have an argument.";;
    esac
done

#echo "isQflag=$isQflag"
#echo "isSflag=$isSflag"
#echo "isDflag=$isDflag"
#echo "isGflag=$isDflag"
if [ ${isQflag} == 0 ]; then
    echo "Warning: q-flag [QUERY_NAME] is empty ${isQflag}. The default q is q2_1."
fi
if [ ${isSflag} == 0 ]; then
    echo "Warning: s-flag [SCALE] is empty. The default s is 0.01."
fi
if [ ${isDflag} == 0 ]; then
    echo "Warning: d-flag [DATABASE] is empty. The default d is systemds."
fi
if [ ${isGflag} == 1 ]; then
    echo "g-flag is set. That means, the docker desktop GUI is used."
fi

echo "Arg 0 (SHELL_SCRIPT): $0"
echo "Arg 1 (QUERY_NAME): ${QUERY_NAME}"
echo "Arg 2 (SCALE): ${SCALE}"
echo "Arg 3 (DB_SYSTEM): ${DB_SYSTEM}"

# Check whether the query is valid.
QUERY_NAME=$(echo "${QUERY_NAME}" | sed 's/\./_/')
isQuery_valid=0
if [ "${QUERY_NAME}" != "all" ]; then
    for q in ${dml_query_array[@]}; do
        if [ ${QUERY_NAME} == ${q} ]; then
            isQuery_valid=1
            break
        fi
    done
    if [ isQuery_valid == 0 ]; then
        echo -e "Sorry, this query ${QUERY_NAME} is invalid. Valid query names are 'all' and ${dml_query_array[@]}."
        echo -e "${RED}Test bench terminated unsuccessfully.${NC}"
        exit
    fi
else
    echo "All queries: ${dml_query_array[@]}"
fi

# Check for the existing required packages. If not install them.
isAllowed="no"
echo "=========="
echo -e "${GREEN}Install required packages${NC}"
echo -e "${GREEN}Check whether the following packages exist:${NC}"
echo "If only SystemDS: docker 'docker compose' git gcc cmake make"
echo "For PostgreSQL: 'docker compose'"
echo "For DuckDB: duckdb"

for package in docker git gcc cmake make; do
    if [ ! "$(${package} --version)" ]; then
        echo "${package} package is required for this test bench. Do you want to allow the installation? (yes/no)"
        read -r isAllowed
        while [ "${isAllowed}" != "yes" ] || [ "${isAllowed}" != "y" ]; do
            if [ "${isAllowed}" == "yes" ] || [ "${isAllowed}" == "y" ]; then
                echo "Your anwser is ${isAllowed}."
                echo "sudo apt-get install ${package}"
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
isAllowed="no"
if [ "${DB_SYSTEM}" != "systemds" ] && [ ! "$(docker compose version)" ]; then
    echo "docker compose is required for this test bench. Do you want to allow the installation? (yes/no)"
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
isAllowed="no"
if ([ "${DB_SYSTEM}" == "duckdb" ] || [ "${DB_SYSTEM}" == "all" ] ) && [ ! "$(duckdb --version)" ]; then
    echo "duckdb is required for this test bench. Do you want to allow the installation? (yes/no)"
    read -r isAllowed
    while [ "${isAllowed}" != "yes" ] || [ "${isAllowed}" != "y" ]; do
        if [ ${isAllowed} == "yes" ]; then
            echo "Your anwser is ${isAllowed}."
            echo "curl https://install.duckdb.org | sh"
            curl https://install.duckdb.org | sh
        elif [ "${isAllowed}" == "no" ] || [ "${isAllowed}" == "n" ]; then
            echo -e "${RED}Sorry, we cannot continue with that test bench without the required packages. The test bench is stopped.${NC}"
            exit
        else
            echo "Your answer '${isAllowed}' is neither 'yes' or 'no'. Please try again."
        fi
        read -r isAllowed
    done
fi

isAllowed="no"
# Use docker desktop GUI
if [ ${isGflag} == 1 ]; then
    if [ ! "$(gnome-terminal --version)" ]; then
        echo "gnome-terminal package is required for this test bench. Do you want to allow the installation? (yes/no)"
        read -r isAllowed
        while [ "${isAllowed}" != "yes" ] || [ "${isAllowed}" != "y" ]; do
            if [ "${isAllowed}" == "yes" ] || [ "${isAllowed}" == "y" ]; then
                echo "Your anwser is ${isAllowed}."
                echo "sudo apt-get install gnome-terminal"
                sudo apt-get install gnome-terminal
            elif [ "${isAllowed}" == "no" ] || [ "${isAllowed}" == "n" ]; then
                echo -e "${RED}Sorry, we cannot continue with that test bench without the required packages. The test bench is stopped.${NC}"
                exit
            else
                echo "Your answer '${isAllowed}' is neither 'yes' or 'no'. Please try again."
                read -r isAllowed
            fi
        done
    fi
    if [ ! "$(docker desktop version)" ]; then
        echo "docker desktop is required for this test bench. Do you want to allow the installation? (yes/no)"
        read -r isAllowed
        while [ "${isAllowed}" != "yes" ] || [ "${isAllowed}" != "y" ]; do
            if [ ${isAllowed} == "yes" ]; then
                echo "Your anwser is ${isAllowed}."
                echo "curl https://install.duckdb.org | sh"
                curl https://install.duckdb.org | sh
            elif [ "${isAllowed}" == "no" ] || [ "${isAllowed}" == "n" ]; then
                echo -e "${RED}Sorry, we cannot continue with that test bench without the required packages. The test bench is stopped.${NC}"
                exit
            else
                echo "Your answer '${isAllowed}' is neither 'yes' or 'no'. Please try again."
            fi
            read -r isAllowed
        done
    fi
fi

# Check whether the data directory exists.
echo "=========="
echo -e "${GREEN}Check for existing data directory and prepare the ssb-dbgen${NC}"
if [ ! -d ssb-dbgen ]; then
    git clone https://github.com/eyalroz/ssb-dbgen.git --depth 1
    cd ssb-dbgen
else
    cd ssb-dbgen
    echo "Can we look for new updates of the datagen repository?. If there are, do you want to pull it? (yes/no)"
    read -r isAllowed
    while [ "${isAllowed}" != "yes" ] || [ "${isAllowed}" != "y" ]; do
            if [ "${isAllowed}" == "yes" ] || [ "${isAllowed}" == "y" ]; then
                echo "Your answer is '${isAllowed}'"
                echo "git pull"
                git pull
            elif [ "${isAllowed}" == "no" ] || [ "${isAllowed}" == "n" ]; then
                echo "Your answer is '${isAllowed}'. No pulls. Use the currently existing version locally."
                break
            else
                echo "Your answer '${isAllowed}' is neither 'yes' or 'no'. Please try again."
                read -r isAllowed
            fi
        done
fi

echo "=========="
echo -e "${GREEN}Build ssb-dbgen and generate data with a given scale factor${NC}"
# Build the generator
cmake -B ./build && cmake --build ./build
# Run the generator (with -s )
build/dbgen -b dists.dss -v -s $SCALE
mkdir -p ../data_dir
mv *.tbl ../data_dir

# Go back to ssb home directory
cd ..
echo "Number of rows of created tables."
for table in customer part supplier date lineorder; do
        str1=`wc --lines < data_dir/${table}.tbl`
        echo "Table ${table} has ${str1} rows." 
done

# Execute queries in SystemDS docker container.
if [ "${DB_SYSTEM}" == "systemds" ] || [ "${DB_SYSTEM}" == "systemds_stats" ] || [ "${DB_SYSTEM}" == "all" ] ; then
    echo "=========="

    echo -e "${GREEN}Start the SystemDS docker container."
    if [ ${isGflag} == 1 ]; then
        docker desktop start
    else
        sudo systemctl start docker
    fi
    
    if [ ! "$(docker images apache/systemds:latest)" ]; then
        docker pull apache/systemds:latest
    fi

    echo "=========="

    echo -e "${GREEN}Execute DML queries in SystemDS${NC}"
    QUERY_NAME=$(echo "${QUERY_NAME}" | sed 's/\./_/')

    docker desktop
    #Enable extended outputs with stats in SystemDs
    useStats=""
    if [ "${DB_SYSTEM}" == "systemds_stats" ]; then
        useStats="--stats"
    fi
    ##all: {"q1_1","q1_2","q1_3","q2_1","q2_2","q2_3","q3_1","q3_2","q3_3","q3_4","q4_1","q4_2","q4_3"} 
    if [ "${QUERY_NAME}" == "all" ]; then
        echo "Execute all 13 queries."
        
        for q in ${dml_query_array[@]} ; do
            echo "Execute query ${q}.dml"
            docker run -it --rm -v $PWD:/scripts/ apache/systemds:latest -f /scripts/queries/${q}.dml ${useStats} -nvargs input_dir="/scripts/data_dir"
        done
    else
        echo "Execute query ${QUERY_NAME}.dml"
        docker run -it --rm -v $PWD:/scripts/ apache/systemds:latest -f /scripts/queries/${QUERY_NAME}.dml ${useStats} -nvargs input_dir="/scripts/data_dir"
    fi
fi

# Execute queries in PostgreSQL docker container.
if [ "${DB_SYSTEM}" == "postgres" ] || [ "${DB_SYSTEM}" == "all" ] ; then
    echo "=========="
    echo -e "${GREEN}Start the PostgreSQL Docker containter and load data.${NC}"

    if [ ${isGflag} == 1 ]; then
        docker desktop start
    else
        sudo systemctl start docker
    fi

    if [ ! "$(docker images postgres:latest)" ]; then
        docker pull postgres:latest
    fi
    
    #Look more in the documentation.
    #https://docs.docker.com/reference/cli/docker/container/ls/
    echo "AM HERE"
    #TO DO solve here.
    if [ "$(docker ps -aq --filter name=${PG_CONTAINER})" ]; then
        if [ ! "$(docker ps -q --filter name=${PG_CONTAINER})" ]; then
            echo "Starting existing container..."
            docker start ${PG_CONTAINER}
        fi
    else
        echo "Creating new PostgreSQL container..."
        echo "$PWD/docker-compose.yaml"
        docker compose -f "$PWD/docker-compose.yaml" up -d --build 
        sleep 3
    fi
    echo "AM HERE2"
    # Load data and copy into the database
    
    for table in customer part supplier date lineorder; do
        #docker exec -i ${PG_CONTAINER} ls
        docker cp data_dir/${table}.tbl ${PG_CONTAINER}:/tmp
        echo "Load ${table} table with number_of_rows:"
        docker exec -i ${PG_CONTAINER} sed -i 's/|$//' "tmp/${table}.tbl"
        docker exec -i ${PG_CONTAINER} psql -U ${POSTGRES_USER} -d ${POSTGRES_DB} -c "TRUNCATE TABLE ${table} CASCADE; COPY ${table} FROM '/tmp/${table}.tbl' DELIMITER '|';"
    done
    # Change query_name e.g. from q1_1 to q1.1
    QUERY_NAME=$(echo "${QUERY_NAME}" | sed 's/_/./')
    echo "=========="
    echo -e "${GREEN}Execute SQL queries in PostgresSQL${NC}"
    #all: {"q1.1","q1.2","q1.3","q2.1","q2.2","q2.3","q3.1","q3.2","q3.3","q3.4","q4.1","q4.2","q4.3"}
    if [ "${QUERY_NAME}" = "all" ]; then
        echo "Execute all 13 queries."
        for q in ${sql_query_array[@]}; do  
            echo "Execute query ${q}.sql"
            echo "docker exec -i ${PG_CONTAINER} psql -U ${POSTGRES_USER} -d ${POSTGRES_DB}  < sql/${QUERY_NAME}.sql"
            docker exec -i ${PG_CONTAINER} psql -U ${POSTGRES_USER} -d ${POSTGRES_DB}  < sql/${q}.sql
       done
    else
        echo "Execute query ${QUERY_NAME}.sql"
        echo "docker exec -i ${PG_CONTAINER} psql -U ${POSTGRES_USER} -d ${POSTGRES_DB}  < sql/${QUERY_NAME}.sql"
        docker exec -i ${PG_CONTAINER} psql -U ${POSTGRES_USER} -d ${POSTGRES_DB}  < sql/${QUERY_NAME}.sql
    fi
fi

# Execute queries in DuckDB locally.
if [ "${DB_SYSTEM}" == "duckdb" ] || [ "${DB_SYSTEM}" == "all" ]; then

    echo "=========="
    echo -e "${GREEN}Start a DuckDB persistent database and load data.${NC}"
    #https://duckdbsnippets.com/snippets/198/run-sql-file-in-duckdb-cli
    # Create a duckdb persistent database file.
    duckdb shell/test_ssb.duckdb < other/ssb_init.sql

    # Load data and copy into the database.
    for table in customer part supplier date lineorder; do
        echo "Load ${table} table"
        duckdb shell/test_ssb.duckdb -c "COPY ${table} FROM 'data_dir/${table}.tbl'; SELECT COUNT(*) AS number_of_rows FROM ${table};" 
    done

#    # Change query_name e.g. from q1_1 to q1.1
    QUERY_NAME=$(echo "${QUERY_NAME}" | sed 's/_/./')
    echo "=========="
    echo -e "${GREEN}Execute SQL queries in DuckDB${NC}"
    #all: {"q1.1","q1.2","q1.3","q2.1","q2.2","q2.3","q3.1","q3.2","q3.3","q3.4","q4.1","q4.2","q4.3"} 
    if [ "${QUERY_NAME}" = "all" ]; then
        echo "Execute all 13 queries."
        for q in ${sql_query_array[@]}; do
            echo "Execute query ${q}.sql"
            duckdb shell/test_ssb.duckdb < sql/${q}.sql
        done
    else
        echo "Execute query ${QUERY_NAME}.sql"
        duckdb shell/test_ssb.duckdb < sql/${QUERY_NAME}.sql
    fi

fi
echo "=========="
echo -e "${GREEN}Test bench finished successfully.${NC}"
