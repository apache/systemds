#!/usr/bin/env bash

# Set properties
systemDS_Home="/home/sfathollahzadeh/Documents/GitHub/systemds"
LOG4JPROP="$systemDS_Home/scripts/perftest/conf/log4j.properties"
jar_file_path="$systemDS_Home/target/SystemDS.jar"
lib_files_path="$systemDS_Home/target/lib/*"
root_data_path="/media/sfathollahzadeh/Windows1/saeedData/GIODataset/flat/aminer"
home_log="/media/sfathollahzadeh/Windows1/saeedData/GIOLog"
sep="_"
ncols=2001
nrows=10001
result_path="SYSDSFrameExperiment"
declare -a  datasets=("csv")

BASE_SCRIPT="time java\
            -Dlog4j.configuration=file:$LOG4JPROP\
            -Xms1g\
            -Xmx15g\
            -cp\
             $jar_file_path:$lib_files_path\
             org.apache.sysds.runtime.iogen.exp.SYSDSFrameExperimentHDFS\
             "

for ro in 1 #2 3 4 5
do
  for d in "${datasets[@]}"; do
    ./resultPath.sh $home_log $d$ro $result_path
    data_file_name="$root_data_path/$d/$d.data"

          schema_file_name="$root_data_path/$d/$d.schema"
          delimiter=","
          SCRIPT="$BASE_SCRIPT\
                  $delimiter\
                  $schema_file_name\
                  $data_file_name\
                  $d\
                  $home_log/benchmark/$result_path/$d$ro.csv\
                  $nrows
          "
#          echo 3 > /proc/sys/vm/drop_caches && sync
#          sleep 20
          time $SCRIPT
  done
done
