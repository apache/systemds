-------------------------------------------------------------------------------
Apache SystemML (incubating)
-------------------------------------------------------------------------------

SystemML is now an Apache Incubator project! Please see the Apache SystemML
(incubating) website at http://systemml.apache.org/ for more information. The
latest project documentation can be found at the SystemML Documentation website
on GitHub at http://apache.github.io/incubator-systemml/.

SystemML is a flexible, scalable machine learning system. SystemML's
distinguishing characteristics are:

  1. Algorithm customizability via R-like and Python-like languages.
  2. Multiple execution modes, including Standalone, Spark Batch, Spark
     MLContext, Hadoop Batch, and JMLC.
  3. Automatic optimization based on data and cluster characteristics to ensure
     both efficiency and scalability.


-------------------------------------------------------------------------------
SystemML in Standalone Mode
-------------------------------------------------------------------------------

Standalone mode can be run on a single machine, allowing data scientists to
develop algorithms locally without need of a distributed cluster. OSX and
Linux users can use the runStandaloneSystemML.sh script to run in Standalone
mode, while Windows users can use the runStandaloneSystemML.bat script.


-------------------------------------------------------------------------------
Hello World Example
-------------------------------------------------------------------------------

The following example will run a "hello world" DML script on SystemML in
Standalone mode.

$ echo 'print("hello world");' > helloworld.dml
$ ./runStandaloneSystemML.sh helloworld.dml


-------------------------------------------------------------------------------
Running SystemML Algorithms
-------------------------------------------------------------------------------

Several existing algorithms can be found in the scripts directory in the
Standalone distribution. In the following example, we first obtain Haberman's
Survival Data Set. We create a metadata file for this data. We create a
types.csv file that describes the type of each column along with a
corresponding metadata file. We then run the Univariate Statistics algorithm
on the data in Standalone mode. The results are output to the
data/univarOut.mtx file.

$ wget -P data/ http://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data
$ echo '{"rows": 306, "cols": 4, "format": "csv"}' > data/haberman.data.mtd
$ echo '1,1,1,2' > data/types.csv
$ echo '{"rows": 1, "cols": 4, "format": "csv"}' > data/types.csv.mtd
$ ./runStandaloneSystemML.sh scripts/algorithms/Univar-Stats.dml -nvargs X=data/haberman.data TYPES=data/types.csv STATS=data/univarOut.mtx

For more information, please see the online SystemML documentation.

