package org.apache.sysds.runtime.compress;

import java.io.Serializable;

/**
 * Generic container for compressed matrix data.
 * Stores the compressed representation along with metadata
 * needed for decompression and size estimation.
 *
 * @author Nirvan C. Udaysingh Jhurree
 */
public class CompressedMatrix implements Serializable {

    private static final long serialVersionUID = 1L;

    private final CompressionType type;
    private final int numRows;
    private final int numCols;
    private final Object compressedData;   // Technique-specific data
    private final double compressionRatio;
    private final byte[] metadata;         // Optional: scaling factors, etc.

    public CompressedMatrix(CompressionType type, int numRows, int numCols,
                            Object compressedData, double compressionRatio) {
        this(type, numRows, numCols, compressedData, compressionRatio, null);
    }

    public CompressedMatrix(CompressionType type, int numRows, int numCols,
                            Object compressedData, double compressionRatio,
                            byte[] metadata) {
        this.type = type;
        this.numRows = numRows;
        this.numCols = numCols;
        this.compressedData = compressedData;
        this.compressionRatio = compressionRatio;
        this.metadata = metadata;
    }

    public CompressionType getType() { return type; }
    public int getNumRows() { return numRows; }
    public int getNumCols() { return numCols; }
    public Object getCompressedData() { return compressedData; }
    public double getCompressionRatio() { return compressionRatio; }
    public byte[] getMetadata() { return metadata; }

    /** Estimate original size in bytes (8 bytes per double) */
    public long estimateOriginalSizeBytes() {
        return (long) numRows * numCols * 8;
    }

    /** Estimate compressed size in bytes */
    public long getCompressedSizeBytes() {
        if (compressedData instanceof byte[]) {
            return ((byte[]) compressedData).length;
        }
        return 0;
    }

    @Override
    public String toString() {
        return String.format("CompressedMatrix[%s, %dx%d, ratio=%.2fx]",
            type.getId(), numRows, numCols, compressionRatio);
    }
}