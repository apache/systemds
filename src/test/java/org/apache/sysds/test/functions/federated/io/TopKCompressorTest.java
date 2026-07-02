/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysds.test.functions.federated.io;

import org.apache.sysds.runtime.controlprogram.federated.compression.CompressedMatrix;
import org.apache.sysds.runtime.controlprogram.federated.compression.CompressionType;
import org.apache.sysds.runtime.controlprogram.federated.compression.TopK.TopKCompressor;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.AutomatedTestBase;
import org.junit.Test;
import static org.junit.Assert.*;

public class TopKCompressorTest extends AutomatedTestBase {

    @Override
    public void setUp() {}

    @Override
    public void tearDown() {}

    // -----------------------------------------------------------------------
    // Basic properties (merged from testCompressionTypeIsTopK,
    // testDimensionsPreservedAfterDecompression, testCompressionRatioIsPositive)
    // -----------------------------------------------------------------------

    @Test
    public void testTopKBasicProperties() throws Exception {
        MatrixBlock input = createRandomMatrix(10, 20);
        TopKCompressor compressor = new TopKCompressor(0.1);
        CompressedMatrix compressed = compressor.compress(input);

        assertEquals(CompressionType.TOPK, compressed.getType());

        MatrixBlock result = compressor.decompress(compressed);
        assertEquals(10, result.getNumRows());
        assertEquals(20, result.getNumColumns());

        assertTrue("Compression ratio must be > 0",
            compressed.getCompressionRatio() > 0);
    }

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

    // -----------------------------------------------------------------------
    // Compression ratio
    // -----------------------------------------------------------------------

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
}
