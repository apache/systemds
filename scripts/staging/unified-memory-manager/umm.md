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

# Unified Memory Manager - Design Document

| **Author(s)** | Philipp Ortner |
:-------------- |:----------------------------------------------------|

## Description
This document describes the initial design of an Unified Memory Manager proposed
for SystemDS.

## Design
The Unified Memory Manager, henceforth UMM, will act as a manager for heap memory
provided by the JVM.

The UMM has a certain (% of JVM heap) amount of memory which it can distribute for operands
and a buffer pool.

Operands are the compiled programs variables which extends the base class 
`Data`, e.g., `MatrixObject` and `StringObject`.
The buffer pool manages
in memory representation of dirty objects that don't exist on the HDFS or 
other persistent memory. This is currently done by the `LazyWriteBuffer`. Intermediates
could be represented in this buffer.

These two memory areas each will have a min and max amount of memory it can
occupy, meaning that the boundary for the areas can shift dynamically depending
on the current load.

||min|max|
| ------------- |:-------------:| -----:|
| operations  | 50% | 70% |
| buffer pool | 15% | 30% |

The UMM will utilise the existing `CacheBlock` structure to manage the buffer
pool area while it will use the new `OperandBlock (?)` structure to keep track of
operations.

### UMM Java Class
For starters, we want to keep the API rather simple.

```java
/**
 * Pins the input block into main memory.
 * The provided block needs to fit into UMM boundaries.
 */
void pin(String, Block);

/**
 * Unpins the input block from main memory.
 */
void unpin(String);

/**
 * Loads the block into memory and pins it, if not evicted just pins it(?)
 * The requested block needs to fit into UMM boundaries.
 */
Block serve(String);

/**
 * Reserves an specified amount in one of the two memory areas.
 * The requested block needs to fit into UMM boundaries.
 */
Block reserve(Size, MemoryArea);

/**
 * Resizes reserved memory in one of the two memory areas.
 * The requested block needs to fit into UMM boundaries.
 */
Block resize_reserved(Size, MemoryArea);

```

The UMM will also needs to keep track of the current state for which it will 
have a few member variables. A queue similar to the current `EvictionQueue` is used
to add/remove entries but rather than having a FIFO eviction policy 
this queue will use LRU.


```java
class UMM {
    // maximum capacity of the UMM
    long _capacity;
    // current operand area size
    long _opSize;
    // current buffer area size
    long _bufferSize;
    // operand block queue
    EvictionQueue _opQueue;
    // buffer block queue
    EvictionQueue _bufferQueue;
    // block handle, maybe a LocalVariableMap(?)
    Map<String, Block> pinnedBlocks;
}
```


### Block Java Class
The first implementation will only handle CacheBlock objects, at the moment
without any major changes to the CacheBlock object itself.

If there is enough time this will be extended to an additional block type, `OperandBlock`.
This would either require a new data type or maybe we could reuse some of the ideas
from the `CacheableData<T extends CacheBlock> extends Data` class.

### Testing
Testing will be done with the existing java based test system which runs a
.dml script on SystemDS.

I suppose this could need some improvments and I would appreciate any help here.

```R
    X = matrix() # random, larger than 15% maxheap, 2*X should fit in main memory budget

    ColMean = colMeans(X) # pins X
    #ouput is reserved

    X =  X - ColMean # pins ColMean, ouput larger than X/15%

    ColSd = ColSd(X) 
    X = X/ColSd(X) # second eviction # unpins old X
    write(X, filename)
```

Additionally, I will also write tests that check if objects are only saved
once in the UMM and don't get duplicated and some smaller tests covering
basic functionality.
