
#!/bin/bash
#Mark as executable.
#chmod +x run_script.sh

#You can run in both modes.
#  ./run_script.sh all [SCALE] # For all queries
#  ./run_script.sh q[QUERY_NUMBER] [SCALE] # For a certain query
#Example
#  ./run_script.sh all 0.1
#  ./run_script.sh q_4_3 0.1
QUERY=$1
SCALE=$2

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Test environment with SSB Data Loader ===${NC}\n"

echo "Arg 1 (QUERY): $0"
echo "Arg 1 (QUERY): $1"
echo "Arg 2 (SCALE): $2"

# Install docker.
echo -e "${GREEN}Install packages${NC}"
echo -e "${BLUE}sudo apt install docker git gcc cmake make${NC}"
sudo apt install docker git gcc cmake make

# Check whether the data directory exists.
#cd ..
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

echo -e "${GREEN}Executing DML queries${NC}"

##all: {"q1_1","q1_2","q1_3","q2_1","q2_2","q2_3","q3_1","q3_2","q3_3","q3_4","q4_1","q4_2","q4_3"} 
if [[ $QUERY = "all" ]]
then
    echo "Execute all 13 queries."
    for q in {"q1_1","q1_2","q1_3","q2_1","q2_2","q2_3","q3_1","q3_2","q3_3","q3_4","q4_1","q4_2","q4_3"}
    do  
        docker run -it --rm -v $PWD:/scripts/ apache/systemds:latest -f /scripts/queries/$q.dml -nvargs input_dir="/scripts/data_dir"
    done
else
    echo "Execute query $QUERY"
    docker run -it --rm -v $PWD:/scripts/ apache/systemds:latest -f /scripts/queries/$QUERY.dml -nvargs input_dir="/scripts/data_dir"
fi
