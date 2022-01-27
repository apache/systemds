#!/usr/bin/env bash

# Set properties
systemDS_Home="/home/sfathollahzadeh/Documents/GitHub/systemds"
LOG4JPROP="$systemDS_Home/scripts/perftest/conf/log4j.properties"
jar_file_path="$systemDS_Home/target/SystemDS.jar"
lib_files_path="$systemDS_Home/target/lib/*"
root_data_path="/media/sfathollahzadeh/Windows1/saeedData/GIODataset/flat/aminer"
home_log="/media/sfathollahzadeh/Windows1/saeedData/FlatDatasets/LOG"
sep="_"
ncols=7
result_path="GIOFrameExperiment"
declare -a  datasets=("aminer_author")

BASE_SCRIPT="time java\
            -Dlog4j.configuration=file:$LOG4JPROP\
            -Xms1g\
            -Xmx15g\
            -cp\
             $jar_file_path:$lib_files_path\
             org.apache.sysds.runtime.iogen.exp.GIOFrameExperimentHDFS\
             "

for ro in 1 #2 3 4 5
do
  for d in "${datasets[@]}"; do
    ./resultPath.sh $home_log $d$ro $result_path
    data_file_name="$root_data_path/$d/$d.data"
    for sr in 100 #20 30 40 50 60 70 80 90 100
      do
        for p in 7
         do
          schema_file_name="$root_data_path/$d/$d$sep$ncols.schema"
          sample_raw_fileName="$root_data_path/$d/sample_$sr$sep$ncols.raw"
          sample_frame_file_name="$root_data_path/$d/sample_$sr$sep$ncols.frame"
          delimiter=","
          SCRIPT="$BASE_SCRIPT\
                  $sample_raw_fileName\
                  $sample_frame_file_name\
                  $sr\
                  $delimiter\
                  $schema_file_name\
                  $data_file_name\
                  $p\
                  $d\
                  $home_log/benchmark/$result_path/$d$ro.csv
          "
#          echo 3 > /proc/sys/vm/drop_caches && sync
#          sleep 20
          #echo $SCRIPT
          time $SCRIPT
        done
      done
  done
done
