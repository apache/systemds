package org.apache.sysds.runtime.compress.TopK;

import org.apache.sysds.runtime.compress.CompressedMatrix;
import org.apache.sysds.runtime.compress.CompressionType;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 * Unit tests for TopKCompressor.
 * Verifies compression ratio, reconstruction accuracy,
 * and correct handling of edge cases.
 *
 *
 */
public class TopKCompressorTest {

    // -----------------------------------------------------------------------
    // Basic compression / decompression
    // -----------------------------------------------------------------------

    @Test
    public void testTopKKeepsLargestElements() throws Exception {
        // 3x3 matrix with three distinct magnitudes
        MatrixBlock input = new MatrixBlock(3, 3, false);
        input.allocateDenseBlock();
        input.set(0, 0, 10.0);  // Largest
        input.set(1, 1, 5.0);   // Medium
        input.set(2, 2, 1.0);   // Smallest
        input.examSparsity();

        // Keep top 2 of 9 elements (~22% sparsity)
        TopKCompressor compressor = new TopKCompressor(0.22);
        CompressedMatrix compressed = compressor.compress(input);
        MatrixBlock result = compressor.decompress(compressed);

        // Largest two values must be preserved exactly
        assertEquals(10.0, result.get(0, 0), 1e-10);
        assertEquals(5.0,  result.get(1, 1), 1e-10);

        // Smallest should be zeroed out
        assertEquals(0.0, result.get(2, 2), 1e-10);
    }

    @Test
    public void testCompressionTypeIsTopK() throws Exception {
        MatrixBlock input = createDenseMatrix(4, 4, 1.0);
        TopKCompressor compressor = new TopKCompressor(0.5);
        CompressedMatrix compressed = compressor.compress(input);
        assertEquals(CompressionType.TOPK, compressed.getType());
    }

    @Test
    public void testDimensionsPreservedAfterDecompression() throws Exception {
        MatrixBlock input = createRandomMatrix(10, 20);
        TopKCompressor compressor = new TopKCompressor(0.1);
        CompressedMatrix compressed = compressor.compress(input);
        MatrixBlock result = compressor.decompress(compressed);

        assertEquals(10, result.getNumRows());
        assertEquals(20, result.getNumColumns());
    }

    // -----------------------------------------------------------------------
    // Compression ratio
    // -----------------------------------------------------------------------

    @Test
    public void testCompressionRatioIsPositive() throws Exception {
        MatrixBlock input = createRandomMatrix(50, 50);
        TopKCompressor compressor = new TopKCompressor(0.01);
        CompressedMatrix compressed = compressor.compress(input);
        assertTrue("Compression ratio must be > 0",
            compressed.getCompressionRatio() > 0);
    }

    @Test
    public void testLowerSparsityGivesHigherRatio() throws Exception {
        MatrixBlock input = createRandomMatrix(100, 100);

        TopKCompressor c1 = new TopKCompressor(0.1);
        TopKCompressor c2 = new TopKCompressor(0.01);

        double ratio1 = c1.compress(input).getCompressionRatio();
        double ratio2 = c2.compress(input).getCompressionRatio();

        assertTrue("1% sparsity should compress more than 10%", ratio2 > ratio1);
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    @Test
    public void testAllZeroMatrix() throws Exception {
        MatrixBlock input = new MatrixBlock(5, 5, false);
        input.allocateDenseBlock();
        input.examSparsity();

        TopKCompressor compressor = new TopKCompressor(0.1);
        CompressedMatrix compressed = compressor.compress(input);
        MatrixBlock result = compressor.decompress(compressed);

        // All zeros in → all zeros out
        for (int i = 0; i < 5; i++)
            for (int j = 0; j < 5; j++)
                assertEquals(0.0, result.get(i, j), 1e-10);
    }

    @Test
    public void testSparsityOfOneKeepsEverything() throws Exception {
        MatrixBlock input = createRandomMatrix(5, 5);
        TopKCompressor compressor = new TopKCompressor(1.0);
        CompressedMatrix compressed = compressor.compress(input);
        MatrixBlock result = compressor.decompress(compressed);

        // With sparsity=1.0, all values should be preserved
        for (int i = 0; i < 5; i++)
            for (int j = 0; j < 5; j++)
                assertEquals(input.get(i, j), result.get(i, j), 1e-10);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testInvalidSparsityThrowsException() {
        new TopKCompressor(0.0);  // Must be > 0
    }

    @Test(expected = IllegalArgumentException.class)
    public void testSparsityAboveOneThrowsException() {
        new TopKCompressor(1.5);  // Must be <= 1
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

    private MatrixBlock createDenseMatrix(int rows, int cols, double fillValue) {
        MatrixBlock m = new MatrixBlock(rows, cols, false);
        m.allocateDenseBlock();
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                m.set(i, j, fillValue);
        m.examSparsity();
        return m;
    }
}