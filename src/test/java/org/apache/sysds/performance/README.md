<!--
{% comment %}
Licensed to the Apache Software Foundation (ASF) under one or more
contributor license agreements.  See the NOTICE file distributed with
this work for additional information regarding copyright ownership.
The ASF licenses this file to you under the Apache License, Version 2.0
(the "License"); you may not use this file except in compliance with
the License.  You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
{% end comment %}
-->

# Performance tests

To compile:

```bash
mvn package
```

Example of running it:

```bash
java -jar target/systemds-3.3.0-SNAPSHOT-perf.jar 1
```

example result of the above job:

```txt
Running Steam Compression Test
      StreamCompress  Repetitions: 100 GenMatrices rand(10000, 100, 32, 1.0) Seed: 42
               In Memory Block Size,    0.010+-  0.010 ms, 8000152.00
                Write Blocks Stream,   16.272+-  4.027 ms, 8000009.00
               Write Stream Deflate,  444.453+- 37.761 ms, 1037452.78
        Write Stream Deflate Speedy,  276.331+- 24.461 ms, 1362222.20
 In Memory Compress Individual (CI),   29.070+-  2.083 ms, 1041744.00
                    Write CI Stream,   31.607+-  1.722 ms, 1027621.00
            Write CI Deflate Stream,   63.826+-  1.693 ms,  655801.16
     Write CI Deflate Stream Speedy,   54.482+-  1.266 ms,  678390.92
```

With profiler:

```bash
java -jar -agentpath:$HOME/Programs/profiler/lib/libasyncProfiler.so=start,event=cpu,file=temp/log.html target/systemds-3.3.0-SNAPSHOT-perf.jar 12 10000 100 4 1.0 16 1000 -1
```

Take a Matrix and perform serialization

```bash 
java -jar -agentpath:$HOME/Programs/profiler/lib/libasyncProfiler.so=start,event=cpu,file=temp/log.html target/systemds-3.3.0-SNAPSHOT-perf.jar 13 16 100 "temp/test.csv" -1
```

Take a Frame and transform into a Matrix and perform serialization.

```bash 
java -jar -agentpath:$HOME/Programs/profiler/lib/libasyncProfiler.so=start,event=cpu,file=temp/log.html target/systemds-3.3.0-SNAPSHOT-perf.jar 14 16 1000 "src/test/resources/datasets/titanic/titanic.csv" "src/test/resources/datasets/titanic/tfspec.json" -1
```

Frame Operation timings

```bash
java -jar -agentpath:$HOME/Programs/profiler/lib/libasyncProfiler.so=start,event=cpu,file=temp/log.html target/systemds-3.3.0-SNAPSHOT-perf.jar 15 16 10 "src/test/resources/datasets/titanic/titanic.csv" "src/test/resources/datasets/titanic/tfspec.json"
```

Reshape Sparse

```bash
java -cp "target/systemds-3.3.0-SNAPSHOT-perf.jar:target/lib/*" -agentpath:$HOME/Programs/profiler/lib/libasyncProfiler.so=start,event=cpu,file=temp/log.html  org.apache.sysds.performance.Main 1005
```


Binary Operations

```bash
java -jar -agentpath:$HOME/Programs/profiler/lib/libasyncProfiler.so=start,event=cpu,file=temp/log.html -XX:+UseNUMA target/systemds-3.3.0-SNAPSHOT-perf.jar 1006 500
```


transform encode 

```bash
java -jar -agentpath:$HOME/Programs/profiler/lib/libasyncProfiler.so=start,event=cpu,file=temp/log.html -XX:+UseNUMA target/systemds-3.3.0-SNAPSHOT-perf.jar 1007
```


append matrix sequence 

```bash
./src/test/scripts/performance/append.sh
```
