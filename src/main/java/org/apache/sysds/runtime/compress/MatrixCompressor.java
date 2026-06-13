package org.apache.sysds.runtime.compress;

import org.apache.sysds.runtime.compress.exceptions.CompressionException;
import org.apache.sysds.runtime.compress.exceptions.DecompressionException;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

/**
 * Interface for matrix compression techniques in federated learning.
 * All compressors must implement compress/decompress operations.
 *
 * @author Nirvan C. Udaysingh Jhurree
 */
public interface MatrixCompressor {

    /**
     * Compress a matrix block for transmission.
     * @param input The source matrix to compress
     * @return CompressedMatrix containing compressed data and metadata
     * @throws CompressionException if compression fails
     */
    CompressedMatrix compress(MatrixBlock input) throws CompressionException;

    /**
     * Decompress a compressed matrix back to MatrixBlock.
     * @param compressed The compressed data to decompress
     * @return Reconstructed MatrixBlock (may be approximate)
     * @throws DecompressionException if decompression fails
     */
    MatrixBlock decompress(CompressedMatrix compressed) throws DecompressionException;

    /**
     * Get the compression technique identifier.
     * @return CompressionType enum value
     */
    CompressionType getCompressionType();

    /**
     * Estimate the compression ratio achieved.
     * Higher is better (e.g. 10.0 means 10x smaller).
     */
    default double estimateCompressionRatio(long originalSize, long compressedSize) {
        return compressedSize == 0 ? Double.MAX_VALUE : (double) originalSize / compressedSize;
    }
}