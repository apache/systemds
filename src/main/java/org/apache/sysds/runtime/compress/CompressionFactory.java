package org.apache.sysds.runtime.compress;

import org.apache.sysds.runtime.compress.TopK.TopKCompressor;
import org.apache.sysds.runtime.compress.Quantization.ProbabilisticQuantizationCompressor;

/**
 * Factory for creating compressor instances from configuration.
 * Centralizes compressor instantiation and parameter validation.
 *
 * Usage:
 *   CompressionConfig config = CompressionConfig.builder()
 *       .enable(true)
 *       .withType(CompressionType.TOPK)
 *       .withSparsity(0.01)
 *       .build();
 *   MatrixCompressor compressor = CompressionFactory.create(config);
 *
 * 
 */
public class CompressionFactory {

    private CompressionFactory() {
        // Utility class — no instantiation
    }

    /**
     * Create a compressor from a CompressionConfig.
     * @param config The compression configuration
     * @return A ready-to-use MatrixCompressor
     * @throws IllegalArgumentException if the config is invalid
     */
    public static MatrixCompressor create(CompressionConfig config) {
        if (config == null || !config.isEnabled()) {
            return new PassthroughCompressor();
        }
        return create(config.getType(), config);
    }

    /**
     * Create a compressor for a specific type with given config.
     */
    public static MatrixCompressor create(CompressionType type, CompressionConfig config) {
        switch (type) {
            case TOPK:
                double sparsity = config.getSparsity();
                return new TopKCompressor(sparsity, true);

            case PROBABILISTIC_QUANTIZATION:
                int bits = config.getBits();
                return new ProbabilisticQuantizationCompressor(bits);

            case ONE_BIT_CS:
                throw new UnsupportedOperationException(
                    "1-Bit Compressed Sensing not yet implemented");

            case NONE:
            default:
                return new PassthroughCompressor();
        }
    }

    // -----------------------------------------------------------------------
    // Passthrough compressor (no-op) for when compression is disabled
    // -----------------------------------------------------------------------

    /**
     * No-op compressor: returns the matrix as-is.
     * Used when compression is disabled or type is NONE.
     */
    private static class PassthroughCompressor implements MatrixCompressor {

        @Override
        public CompressedMatrix compress(org.apache.sysds.runtime.matrix.data.MatrixBlock input)
                throws org.apache.sysds.runtime.compress.exceptions.CompressionException {
            return new CompressedMatrix(
                CompressionType.NONE,
                input.getNumRows(),
                input.getNumColumns(),
                input,
                1.0
            );
        }

        @Override
        public org.apache.sysds.runtime.matrix.data.MatrixBlock decompress(CompressedMatrix compressed)
                throws org.apache.sysds.runtime.compress.exceptions.DecompressionException {
            return (org.apache.sysds.runtime.matrix.data.MatrixBlock) compressed.getCompressedData();
        }

        @Override
        public CompressionType getCompressionType() {
            return CompressionType.NONE;
        }
    }
}