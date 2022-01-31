package org.apache.sysds.runtime.matrix.operators;

public enum CountDistinctOperatorTypes { // The different supported types of counting.
    COUNT, // Baseline naive implementation, iterate through, add to hashMap.
    KMV, // K-Minimum Values algorithm.
    HLL // HyperLogLog algorithm.
}
