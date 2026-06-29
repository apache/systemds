package org.apache.sysds.runtime.compress.Quantization;

import org.apache.sysds.runtime.compress.CompressedMatrix;
import org.apache.sysds.runtime.compress.CompressionType;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 * Unit tests for ProbabilisticQuantizationCompressor.
 * Verifies compression ratio, reconstruction accuracy,
 * unbiasedness property, and edge case handling.
 *
 *
 */
public class ProbabilisticQuantizationCompressorTest {

    // -----------------------------------------------------------------------
    // Basic compression / decompression
    // -----------------------------------------------------------------------

    @Test
    public void testCompressionTypeIsProbabilisticQuantization() throws Exception {
        MatrixBlock input = createRandomMatrix(4, 4);
        ProbabilisticQuantizationCompressor compressor =
            new ProbabilisticQuantizationCompressor(4);
        CompressedMatrix compressed = compressor.compress(input);
        assertEquals(CompressionType.PROBABILISTIC_QUANTIZATION, compressed.getType());
    }

    @Test
    public void testDimensionsPreservedAfterDecompression() throws Exception {
        MatrixBlock input = createRandomMatrix(10, 20);
        ProbabilisticQuantizationCompressor compressor =
            new ProbabilisticQuantizationCompressor(4);
        CompressedMatrix compressed = compressor.compress(input);
        MatrixBlock result = compressor.decompress(compressed);

        assertEquals(10, result.getNumRows());
        assertEquals(20, result.getNumColumns());
    }

    @Test
    public void testReconstructedValuesWithinOriginalRange() throws Exception {
        MatrixBlock input = createRandomMatrix(20, 20);

        // Find original min/max
        double origMin = Double.MAX_VALUE;
        double origMax = -Double.MAX_VALUE;
        for (int i = 0; i < 20; i++) {
            for (int j = 0; j < 20; j++) {
                double v = input.get(i, j);
                if (v < origMin) origMin = v;
                if (v > origMax) origMax = v;
            }
        }

        ProbabilisticQuantizationCompressor compressor =
            new ProbabilisticQuantizationCompressor(4);
        MatrixBlock result = compressor.decompress(compressor.compress(input));

        // All reconstructed values must be within [min, max]
        for (int i = 0; i < 20; i++) {
            for (int j = 0; j < 20; j++) {
                double v = result.get(i, j);
                assertTrue("Value below min: " + v, v >= origMin - 1e-9);
                assertTrue("Value above max: " + v, v <= origMax + 1e-9);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Compression ratio
    // -----------------------------------------------------------------------

    @Test
    public void testCompressionRatio2Bit() throws Exception {
        MatrixBlock input = createRandomMatrix(10, 10);
        ProbabilisticQuantizationCompressor compressor =
            new ProbabilisticQuantizationCompressor(2);
        CompressedMatrix compressed = compressor.compress(input);
        assertEquals(16.0, compressed.getCompressionRatio(), 1e-10);
    }

    @Test
    public void testCompressionRatio4Bit() throws Exception {
        MatrixBlock input = createRandomMatrix(10, 10);
        ProbabilisticQuantizationCompressor compressor =
            new ProbabilisticQuantizationCompressor(4);
        CompressedMatrix compressed = compressor.compress(input);
        assertEquals(8.0, compressed.getCompressionRatio(), 1e-10);
    }

    @Test
    public void testCompressionRatio8Bit() throws Exception {
        MatrixBlock input = createRandomMatrix(10, 10);
        ProbabilisticQuantizationCompressor compressor =
            new ProbabilisticQuantizationCompressor(8);
        CompressedMatrix compressed = compressor.compress(input);
        assertEquals(4.0, compressed.getCompressionRatio(), 1e-10);
    }

    @Test
    public void testFewerBitsGivesHigherRatio() throws Exception {
        MatrixBlock input = createRandomMatrix(20, 20);

        double ratio2bit = new ProbabilisticQuantizationCompressor(2)
            .compress(input).getCompressionRatio();
        double ratio8bit = new ProbabilisticQuantizationCompressor(8)
            .compress(input).getCompressionRatio();

        assertTrue("2-bit should compress more than 8-bit", ratio2bit > ratio8bit);
    }

    // -----------------------------------------------------------------------
    // Unbiasedness: E[quantized] ≈ original over many runs
    // -----------------------------------------------------------------------

    @Test
    public void testUnbiasednessOverManyRuns() throws Exception {
        // Single element matrix with value 0.5 (midpoint)
        MatrixBlock input = new MatrixBlock(1, 1, false);
        input.allocateDenseBlock();
        input.set(0, 0, 5.0);
        input.examSparsity();

        // Run quantization 1000 times and average the results
        double sum = 0.0;
        int runs = 1000;
        for (int r = 0; r < runs; r++) {
            ProbabilisticQuantizationCompressor compressor =
                new ProbabilisticQuantizationCompressor(4);
            MatrixBlock result = compressor.decompress(compressor.compress(input));
            sum += result.get(0, 0);
        }
        double average = sum / runs;

        // Average should be close to original value (unbiased estimator)
        assertEquals("Quantization should be unbiased",
            5.0, average, 0.5);  // Allow 0.5 tolerance
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    @Test
    public void testConstantMatrix() throws Exception {
        // All same value — min == max edge case
        MatrixBlock input = new MatrixBlock(3, 3, false);
        input.allocateDenseBlock();
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                input.set(i, j, 7.0);
        input.examSparsity();

        ProbabilisticQuantizationCompressor compressor =
            new ProbabilisticQuantizationCompressor(4);
        CompressedMatrix compressed = compressor.compress(input);
        MatrixBlock result = compressor.decompress(compressed);

        // Should not throw; all values should reconstruct to min (7.0 or close)
        assertNotNull(result);
        assertEquals(3, result.getNumRows());
        assertEquals(3, result.getNumColumns());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testInvalidBitsThrowsException() {
        new ProbabilisticQuantizationCompressor(3);  // Only 2, 4, 8 allowed
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    private MatrixBlock createRandomMatrix(int rows, int cols) {
        MatrixBlock m = new MatrixBlock(rows, cols, false);
        m.allocateDenseBlock();
        java.util.Random rng = new java.util.Random(42);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                m.set(i, j, rng.nextGaussian() * 10);
        m.examSparsity();
        return m;
    }
}