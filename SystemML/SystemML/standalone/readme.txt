SystemML enables declarative, large-scale machine learning (ML) via a high-level language with R-like syntax. Data scientists use this language to express their ML algorithms with full flexibility but without the need to hand-tune distributed runtime execution plans and system configurations. These ML programs are dynamically compiled and optimized based on data and cluster characteristics using rule and cost-based optimization techniques. The compiler automatically generates hybrid runtime execution plans ranging from in-memory, single node execution to distributed MapReduce (MR) computation and data access.

jSystemML.jar is derived out of SystemML.jar, to work in non-Hadoop desktop/laptop environment just like a Java appln. 

We recommend to use "-exec singlenode" option, in order to force in-memory computation.

If you see error "java.lang.OutOfMemoryError", then edit the invocation script to adjust JVM memory "-Xmx20g -Xms20g -Xmn1g".

Please see the help/usage page at :-
java -jar jSystemML.jar -help
java -jar jSystemML.jar -?