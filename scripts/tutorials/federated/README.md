# Federated SystemDS tutorial

This tutorial is dedicated to a distributed example of systemds federated.

## Step 1: Setup Parameters

Before you begin look trough the parameters.sh file, and change the variables to fit your needs.

The default parameters are set to execute a four worker setup on localhost.
If you have access to other machines simply change the address list to the remote locations, either using IP addresses, or aliases.

Also note the memory settings, and set these appropriately

Before going further it is expected that you have setup the default install of SystemDS described in: <http://apache.github.io/systemds/site/install>

## Step 2: Install

This install script setup a local python environment installing systemds and
for all the addresses listed in the address list download and build systemds for both java execution but also for python systemds.

```sh
./install.sh
```

at the time of writing it results in:

```txt
Me:~/github/federatedTutorial$ ./install.sh
Creating Python Virtual Enviroment on XPS-15-7590
Successfully installed certifi-2020.12.5 chardet-4.0.0 idna-2.10 numpy-1.20.3 pandas-1.2.4 py4j-0.10.9.2 python-dateutil-2.8.1 pytz-2021.1 requests-2.25.1 six-1.16.0 systemds-2.1.0 urllib3-1.26.4
Installed Python Systemds
```

Also required is to install NodeJS <https://nodejs.org/en/>, simply download and add the bin folder to path add it on path to enable:

```sh
node --version
```

```txt
Me:~/.../scripts/tutorials/federated$ node --version
v16.17.1
```

## Step 3: Setup and Download Data

Next we download and split the dataset into partitions that the different federated workers can use.

```sh
./setup.sh
```

The expected output is:

```txt
Me:~/github/federatedTutorial$ ./setup.sh
Generating data/mnist_features_2_1.data
SystemDS Statistics:
Total execution time:           0.672 sec.

Generating data/mnist_labels_2_1.data
SystemDS Statistics:
Total execution time:           0.109 sec.
```

and the data folder should contain the following:

```txt
Me:~/github/federatedTutorial$ ls data
fed_mnist_features_1.json      fed_mnist_labels_1.json.mtd  mnist_features_2_1.data      mnist_features.data.mtd    mnist_labels_2_2.data      mnist_test_features.data.mtd
fed_mnist_features_1.json.mtd  fed_mnist_labels_2.json      mnist_features_2_1.data.mtd  mnist_labels_1_1.data      mnist_labels_2_2.data.mtd  mnist_test_labels.data
fed_mnist_features_2.json      fed_mnist_labels_2.json.mtd  mnist_features_2_2.data      mnist_labels_1_1.data.mtd  mnist_labels.data          mnist_test_labels.data.mtd
fed_mnist_features_2.json.mtd  mnist_features_1_1.data      mnist_features_2_2.data.mtd  mnist_labels_2_1.data      mnist_labels.data.mtd
fed_mnist_labels_1.json        mnist_features_1_1.data.mtd  mnist_features.data          mnist_labels_2_1.data.mtd  mnist_test_features.data
```

## Step 4: Start Workers

Now everything is setup, simply start the workers using the startAllWorkers script.

```sh
./startAllWorkers.sh
```

output:

```txt
Me:~/github/federatedTutorial$ ./startAllWorkers.sh
Starting Workers.
Starting worker XPS-15-7590 8002 def
Starting worker XPS-15-7590 8001 def
```

The workers will start and some temporary files will be created containing the PID for the worker, to enable specific termination of the worker after experimentation is done. Note that you can run the algorithm multiple times with the same workers.

```txt
Me:~/github/federatedTutorial$ ls tmp/worker/
XPS-15-7590-8001  XPS-15-7590-8002
Me:~/github/federatedTutorial$ cat tmp/worker/XPS-15-7590-8001
13850
```

Also worth noting is that all the output from the federated worker is concatenated to: results/fed/workerlog/

At startup a federated monitoring tool is launched at <http://localhost:4200>.

## Step 5: Port Forward

if you don't have access to the ports on the federated workers (note local execution can skip this.)

If the ports are not accessible directly from your machine because of a firewall, i suggest using the port forwarding script.
that port forward the list of ports from your local machine to the remote machines.
Note this only works if all the federated machines are remote machines, aka the address list contain no localhost.

```sh
portforward.sh
```

Note these process will just continue running in the background so have to be manually terminated.

## Step 6: run algorithms

This tutorial is using different scripts depending on what is outcommented in the run.sh.

Please go into this file and enable which specific script you want to run.

To execute it simply use:

```sh
./run.sh
```

This have execute three different execution versions.
first with one federated worker, then two and finally a local baseline.

all outputs are put into the results folder and could look like:

```txt
Me:~/github/federatedTutorial$ cat results/fed1/lm_mnist_XPS-15-7590_def.log
SystemDS Statistics:
Total elapsed time:             1.489 sec.
Total compilation time:         0.533 sec.
Total execution time:           0.956 sec.
Cache hits (Mem/Li/WB/FS/HDFS): 6/0/0/0/0.
Cache writes (Li/WB/FS/HDFS):   0/2/0/1.
Cache times (ACQr/m, RLS, EXP): 0.000/0.000/0.001/0.036 sec.
HOP DAGs recompiled (PRED, SB): 0/0.
HOP DAGs recompile time:        0.000 sec.
Federated I/O (Read, Put, Get): 2/0/2.
Federated Execute (Inst, UDF):  6/0.
Total JIT compile time:         1.406 sec.
Total JVM GC count:             2.
Total JVM GC time:              0.023 sec.
Heavy hitter instructions:
  #  Instruction  Time(s)  Count
  1  fed_tsmm       0.637      1
  2  solve          0.231      1
  3  write          0.036      1
  4  +              0.019      1
  5  fed_r'         0.017      1
  6  fed_ba+*       0.010      1
  7  <=             0.007      1
  8  rdiag          0.003      1
  9  rand           0.001      1
 10  rmvar          0.001      6
 11  createvar      0.000     10
 12  mvvar          0.000     26
 13  -              0.000     12
 14  r'             0.000      1
 15  ==             0.000      8
 16  ^              0.000      1
 17  >              0.000      2
 18  ||             0.000      2

real 4,44
user 6,77
sys 0,28
```

Similarly one can see what the federated sites executed in the federated output logs results/fed/workerlog:

```txt
Me:~/github/federatedTutorial$ cat results/fed/workerlog/XPS-15-7590-8002.out
Federated Worker SystemDS Statistics:
Total elapsed time:             0.000 sec.
Total compilation time:         0.000 sec.
Total execution time:           0.000 sec.
Number of compiled Spark inst:  0.
Number of executed Spark inst:  0.
Cache hits (Mem/Li/WB/FS/HDFS): 4/0/0/0/2.
Cache writes (Li/WB/FS/HDFS):   0/0/0/0.
Cache times (ACQr/m, RLS, EXP): 0.287/0.000/0.000/0.000 sec.
HOP DAGs recompiled (PRED, SB): 0/0.
HOP DAGs recompile time:        0.000 sec.
Spark ctx create time (lazy):   0.000 sec.
Spark trans counts (par,bc,col):0/0/0.
Spark trans times (par,bc,col): 0.000/0.000/0.000 secs.
Total JIT compile time:         1.943 sec.
Total JVM GC count:             2.
Total JVM GC time:              0.023 sec.
Heavy hitter instructions:
 #  Instruction  Time(s)  Count
 1  tsmm           0.387      1
 2  r'             0.022      1
 3  ba+*           0.004      1
 4  rmvar          0.000      3
```

## Step 6: Stop Workers

To stop the workers running simply use the stop all workers script.

```sh
./stopAllWorkers.sh
```

```txt
Me:~/.../scripts/tutorials/federated$ ./stopAllWorkers.sh 
Stopping workers XPS-15-7590
Stopping Monitoring 
STOP NPM manually!! Process ID:
62870q
```

As specified by the output npm is still running and have to be manually stopped.
Simply search for 'ng serve' process in the terminal and stop it.
