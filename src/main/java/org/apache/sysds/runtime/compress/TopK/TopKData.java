package org.apache.sysds.runtime.compress.TopK;

import java.io.Serializable;

/**
 * Immutable container for TopK-compressed matrix data.
 * Stores only the K largest-magnitude elements with their positions,
 * designed for efficient serialization across federated workers.
 *
 * @author Nirvan C. Udaysingh Jhurree
 */
public class TopKData implements Serializable {

    private static final long serialVersionUID = 1L;

    public final int[] indices;    // Linear indices of kept elements (row*numCols + col)
    public final double[] values;  // Corresponding original values
    public final int numCols;      // Needed for index → (row, col) conversion

    public TopKData(int[] indices, double[] values, int numCols) {
        if (indices.length != values.length) {
            throw new IllegalArgumentException(
                "Indices and values arrays must have the same length");
        }
        this.indices = indices.clone();  // Defensive copy
        this.values = values.clone();
        this.numCols = numCols;
    }

    /** Number of kept elements */
    public int size() {
        return indices.length;
    }

    /** Estimate serialized size in bytes (4 bytes per int + 8 bytes per double) */
    public long estimateSizeBytes() {
        return (long) indices.length * 12 + 64;  // +64 for object headers
    }

    @Override
    public String toString() {
        return String.format("TopKData[k=%d, numCols=%d]", indices.length, numCols);
    }
}