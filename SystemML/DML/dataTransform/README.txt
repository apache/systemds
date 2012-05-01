/**********************************************
 *
 * Data Transformation:
 *
 * contains script to pre-process the input data
 * to generate input matrix for SystemML
 *
 * to run it against your input
 *
 * 1. first compile metaDataParse.jave and generate the jar file
 *
 *  1) javac -classpath /opt/ibm/biginsights/jaql/jaql.jar:/opt/ibm/biginsights/lib/antlr.jar 
 *           -d classes metaDataParse.java
 *  2) jar -cvf metaDataParse.jar -C classes/ .
 *
 * 2. copy input file into hdfs
 * e.g: hdoop fs -put inputData.txt ./data/inputData.txt
 *
 * 3. prepare the meta data file to be used for processing the data
 *    see detail in sample metaData.json file.
 *
 * 4. run the jaql script (dataTransform.jaql, please note commonFunc.jaql
 *    should reside in the same directory as dataTransform.jaql, will be
 *    imported when running dataTransform.jaql)
 *
 *  e.g: jaqlshell -jp <path-to-jaql-script>  -j <path>/metaDataParse.jar
 *                 -e "metaFile='<path>/metaData.json'" dataTransform.jaql
 *************************************************************************/
 