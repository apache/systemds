
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

# SIMD DoubleVectors for matrix multiplication

`DoubleVector` is still in incubator stage, but promises performance improvements for many SystemDS components.
This patch explored potential speedup for matrix multiplication of two dense matrices. Additionally, dot product
is also implemented with `DoubleVector` for the case where common dimension is `1`.

Initial experiments showed varying results, usually the vectorized implementation performs somewhere between
`MKL` and our reference. There are also cases where we are slower than the reference, or faster than `MKL`.
For detailed discussion (and plots) see PR #1643.

## Further Work

This patch focused only on dense matrix multiplication, increasing sparsity would complicate things.
The sparsity aware copying (see `LibMatrixMult.java:1170`) and general loop structure is kept as it is, as a lot of 
experimentation went into a very efficient implementation. Note that the usage of `DoubleVector` might change
a lot of things about this and revisiting this (and using SIMD for sparsity aware copying) will be a necessary step.

## Changes

Due to the dependency of at least JDK17, there are changes to `pom.xml`, run script `systemds` and, of course, `LibMatrixMult.java`.