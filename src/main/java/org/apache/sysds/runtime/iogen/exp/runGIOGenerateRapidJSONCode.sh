#!/usr/bin/env bash

# Set properties
systemDS_Home="/home/sfathollahzadeh/Documents/GitHub/systemds"
src_cpp="/home/sfathollahzadeh/Documents/GitHub/papers/2022-icde-gIO/experiments/benchmark/RapidJSONCPP/src/at/tugraz"
LOG4JPROP="$systemDS_Home/scripts/perftest/conf/log4j.properties"
jar_file_path="$systemDS_Home/target/SystemDS.jar"
lib_files_path="$systemDS_Home/target/lib/*"
root_data_path="/media/sfathollahzadeh/Windows1/saeedData/NestedDatasets"
home_log="/media/sfathollahzadeh/Windows1/saeedData/NestedDatasets/LOG"
sep="_"
result_path="GIONestedExperiment"
declare -a  datasets=("aminer")

BASE_SCRIPT="time java\
            -Dlog4j.configuration=file:$LOG4JPROP\
            -Xms1g\
            -Xmx15g\
            -cp\
             $jar_file_path:$lib_files_path\
             org.apache.sysds.runtime.iogen.exp.GIOGenerateRapidJSONCode\
             "

for ro in 1 #2 3 4 5
do
  for d in "${datasets[@]}"; do
    ./resultPath.sh $home_log $d$ro $result_path
    for sr in 100
      do
        for p in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
         do
          schema_file_name="$root_data_path/$d/$d$sep$p.schema"
          sample_raw_fileName="$root_data_path/$d/sample_$sr$sep$p.raw"
          sample_frame_file_name="$root_data_path/$d/sample_$sr$sep$p.frame"
          delimiter="\t"
          SCRIPT="$BASE_SCRIPT\
                  $sample_raw_fileName\
                  $sample_frame_file_name\
                  $sr\
                  $delimiter\
                  $schema_file_name\
                  $src_cpp\
                  $p\
                  $d\
                  $home_log/benchmark/$result_path/$d$ro.csv
          "
#          echo 3 > /proc/sys/vm/drop_caches && sync
#          sleep 20
          time $SCRIPT
        done
      done
  done
done
