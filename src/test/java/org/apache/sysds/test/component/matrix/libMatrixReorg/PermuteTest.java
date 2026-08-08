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

package org.apache.sysds.test.component.matrix.libMatrixReorg;

import org.junit.Assert;
import org.junit.Ignore;
import org.junit.Test;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.data.DenseBlock;
import org.mockito.Mockito;
import java.util.Arrays;

public class PermuteTest {

	@Test
	public void testBasicPermuteDense() {
		int[] shape = {2, 3, 4};
		MatrixBlock tensor = generateDenseMatrixBlock(shape);

		Assert.assertEquals(24, tensor.getNumRows() * tensor.getNumColumns());
		
		double[] data = tensor.getDenseBlockValues();
		Assert.assertEquals(23.0, data[1 * 4 * 3 + 2 * 4 + 3], 0.001);
		Assert.assertEquals(0.0, data[0 * 4 * 3 + 0 * 4 + 0], 0.001);

		int[] permutation = {1, 0, 2};
		MatrixBlock outTensor = LibMatrixReorg.permute(tensor, shape, permutation); 

		double[] outData = outTensor.getDenseBlockValues();
		Assert.assertEquals(24, outData.length); 
		Assert.assertEquals(4.0, outData[8], 0.001);
		Assert.assertEquals(15.0, outData[7], 0.001);
	}

	@Test
	public void testPermute2DTransposeDense() {
		int[] shape = {10, 5};
		int[] perm = {1, 0};

		MatrixBlock in = generateDenseMatrixBlock(shape);
		MatrixBlock out = LibMatrixReorg.permute(in, shape, perm);

		verifyPermutation(in, out, shape, perm);
	}

	@Test
	public void testPermute3DSimpleDense() {
		int[] shape = {2, 3, 4};
		int[] perm = {1, 0, 2};

		MatrixBlock in = generateDenseMatrixBlock(shape);
		MatrixBlock out = LibMatrixReorg.permute(in, shape, perm);

		verifyPermutation(in, out, shape, perm);
	}

	@Test
	public void testPermute3DIdentityDense() {
		int[] shape = {5, 5, 5};
		int[] perm = {0, 1, 2};

		MatrixBlock in = generateDenseMatrixBlock(shape);
		MatrixBlock out = LibMatrixReorg.permute(in, shape, perm);

		verifyPermutation(in, out, shape, perm);
	}

	@Test
	public void testPermute4DReverseDense() {
		int[] shape = {2, 3, 4, 5};
		int[] perm = {3, 2, 1, 0};

		MatrixBlock in = generateDenseMatrixBlock(shape);
		MatrixBlock out = LibMatrixReorg.permute(in, shape, perm);

		verifyPermutation(in, out, shape, perm);
	}

	@Test
	public void testPermuteHighRankDense() {
		int[] shape = {2, 2, 2, 2, 2, 2};
		int[] perm = {5, 0, 4, 1, 3, 2};

		MatrixBlock in = generateDenseMatrixBlock(shape);
		MatrixBlock out = LibMatrixReorg.permute(in, shape, perm);

		verifyPermutation(in, out, shape, perm);
	}

	@Test
	public void testLargeBlockLogicMockedDense() {
		int[] shape = {10, 10, 10};
		int[] perm = {2, 0, 1};

		MatrixBlock in = generateDenseMatrixBlock(shape);
		DenseBlock originalDB = in.getDenseBlock();
		DenseBlock spyDB = Mockito.spy(originalDB);
		Mockito.when(spyDB.numBlocks()).thenReturn(2);
		in.setDenseBlock(spyDB);

		MatrixBlock out = LibMatrixReorg.permute(in, shape, perm);

		MatrixBlock reference = generateDenseMatrixBlock(shape);
		verifyPermutation(reference, out, shape, perm);
	}

	@Test
	public void testLargeBlockLogicMockedInputAndOutputDense() {
		int[] shape = {4, 4, 4};
		int[] perm = {2, 1, 0};

		MatrixBlock in = generateDenseMatrixBlock(shape);
		DenseBlock spyIn = Mockito.spy(in.getDenseBlock());
		Mockito.when(spyIn.numBlocks()).thenReturn(5);
		in.setDenseBlock(spyIn);

		MatrixBlock out = LibMatrixReorg.permute(in, shape, perm);

		MatrixBlock reference = generateDenseMatrixBlock(shape);
		verifyPermutation(reference, out, shape, perm);
	}

	@Test
	public void testPermute3DParallelDense() {
		int[] shape = {100, 100, 100};
		int[] perm = {2, 0, 1};

		MatrixBlock in = generateDenseMatrixBlock(shape);
		MatrixBlock out = LibMatrixReorg.permute(in, shape, perm, -1);

		verifyPermutation(in, out, shape, perm);
	}

	@Test
	public void testEdgeCaseSingleElementDense() {
		int[] shape = {1, 1, 1};
		int[] perm = {2, 1, 0};

		MatrixBlock in = generateDenseMatrixBlock(shape);
		MatrixBlock out = LibMatrixReorg.permute(in, shape, perm);

		verifyPermutation(in, out, shape, perm);
	}

	@Test
	public void testEdgeCaseOneDimensionOneDense() {
		int[] shape = {5, 1, 10};
		int[] perm = {2, 0, 1};

		MatrixBlock in = generateDenseMatrixBlock(shape);
		MatrixBlock out = LibMatrixReorg.permute(in, shape, perm);

		verifyPermutation(in, out, shape, perm);
	}

	@Test
	public void testEdgeCaseTwoDimensionsOneDense() {
		int[] shape = {1, 1, 100};
		int[] perm = {2, 1, 0};

		MatrixBlock in = generateDenseMatrixBlock(shape);
		MatrixBlock out = LibMatrixReorg.permute(in, shape, perm);

		verifyPermutation(in, out, shape, perm);
	}

	@Test
	public void testConsecutivePermutationsDense() {
		int[] shape = {3, 4, 5};
		int[] perm1 = {1, 0, 2};
		int[] perm2 = {2, 0, 1};

		MatrixBlock in = generateDenseMatrixBlock(shape);
		MatrixBlock temp = LibMatrixReorg.permute(in, shape, perm1);

		int[] tempShape = {shape[perm1[0]], shape[perm1[1]], shape[perm1[2]]};
		MatrixBlock out = LibMatrixReorg.permute(temp, tempShape, perm2);

		verifyPermutation(temp, out, tempShape, perm2);
	}

	@Test
	public void testDifferentThreadCountsDense() {
		int[] shape = {50, 50, 50};
		int[] perm = {2, 0, 1};

		MatrixBlock in = generateDenseMatrixBlock(shape);

		MatrixBlock out1 = LibMatrixReorg.permute(in, shape, perm, 1);
		MatrixBlock out2 = LibMatrixReorg.permute(in, shape, perm, 2);
		MatrixBlock out4 = LibMatrixReorg.permute(in, shape, perm, 4);
		MatrixBlock out8 = LibMatrixReorg.permute(in, shape, perm, 8);

		double[] data1 = out1.getDenseBlockValues();
		double[] data2 = out2.getDenseBlockValues();
		double[] data4 = out4.getDenseBlockValues();
		double[] data8 = out8.getDenseBlockValues();

		for (int i = 0; i < data1.length; i++) {
			Assert.assertEquals(data1[i], data2[i], 0.0001);
			Assert.assertEquals(data1[i], data4[i], 0.0001);
			Assert.assertEquals(data1[i], data8[i], 0.0001);
		}
	}

	@Test
	public void testThreadCountExceedsElementCountDense() {
		int[] shape = {2, 2};
		int[] perm = {1, 0};
		MatrixBlock in = generateDenseMatrixBlock(shape);

		MatrixBlock out1 = LibMatrixReorg.permute(in, shape, perm, 1);
		MatrixBlock out16 = LibMatrixReorg.permute(in, shape, perm, 16);

		verifyPermutation(in, out16, shape, perm);
		double[] data1 = out1.getDenseBlockValues();
		double[] data16 = out16.getDenseBlockValues();
		for (int i = 0; i < data1.length; i++)
			Assert.assertEquals(data1[i], data16[i], 0.0001);
	}

	@Test
	public void testPermuteAllDimensionsCyclicDense() {
		int[] shape = {3, 4, 5, 2};
		int[] perm = {1, 2, 3, 0};

		MatrixBlock in = generateDenseMatrixBlock(shape);
		MatrixBlock out = LibMatrixReorg.permute(in, shape, perm);

		verifyPermutation(in, out, shape, perm);
	}

	@Test
	public void testPermuteNonContiguousStridesDense() {
		int[] shape = {7, 11, 13};
		int[] perm = {2, 0, 1};

		MatrixBlock in = generateDenseMatrixBlock(shape);
		MatrixBlock out = LibMatrixReorg.permute(in, shape, perm);

		verifyPermutation(in, out, shape, perm);
	}

	@Test
	public void testPermuteLargePrimeStridesDense() {
		int[] shape = {17, 19};
		int[] perm = {1, 0};

		MatrixBlock in = generateDenseMatrixBlock(shape);
		MatrixBlock out = LibMatrixReorg.permute(in, shape, perm);

		verifyPermutation(in, out, shape, perm);
	}

	@Test
	public void testSpecialFloatingPointValuesDense() {
		int[] shape = {2, 2};
		int[] perm = {1, 0};

		MatrixBlock in = new MatrixBlock(1, 4, false);
		in.allocateDenseBlock();
		double[] data = in.getDenseBlockValues();
		data[0] = Double.NaN;
		data[1] = Double.POSITIVE_INFINITY;
		data[2] = Double.NEGATIVE_INFINITY;
		data[3] = 0.0;

		MatrixBlock out = LibMatrixReorg.permute(in, shape, perm);
		double[] outData = out.getDenseBlockValues();

		int nanCount = 0, posInfCount = 0, negInfCount = 0;
		for (double v : outData) {
			if (Double.isNaN(v)) nanCount++;
			else if (v == Double.POSITIVE_INFINITY) posInfCount++;
			else if (v == Double.NEGATIVE_INFINITY) negInfCount++;
		}
		Assert.assertEquals(1, nanCount);
		Assert.assertEquals(1, posInfCount);
		Assert.assertEquals(1, negInfCount);
	}

	@Test
	public void testInputImmutableAfterPermuteDense() {
		int[] shape = {4, 5, 6};
		int[] perm = {2, 0, 1};

		MatrixBlock in = generateDenseMatrixBlock(shape);
		double[] before = in.getDenseBlockValues().clone();

		LibMatrixReorg.permute(in, shape, perm);

		double[] after = in.getDenseBlockValues();
		Assert.assertArrayEquals(before, after, 0.0001);
	}

	@Test
	public void testBasicPermuteSparse() {
		int[] shape = {2, 3, 4};
		int[] perm = {1, 0, 2};

		MatrixBlock in = generateSparseMatrixBlock(shape, 0.2);
		MatrixBlock out = LibMatrixReorg.permute(in, shape, perm);

		verifyPermutation(in, out, shape, perm);
	}

	@Test
	public void testPermute2DTransposeSparse() {
		int[] shape = {10, 5};
		int[] perm = {1, 0};

		MatrixBlock in = generateSparseMatrixBlock(shape, 0.2);
		MatrixBlock out = LibMatrixReorg.permute(in, shape, perm);

		verifyPermutation(in, out, shape, perm);
	}

	@Test
	public void testPermute3DSimpleSparse() {
		int[] shape = {2, 3, 4};
		int[] perm = {1, 0, 2};

		MatrixBlock in = generateSparseMatrixBlock(shape, 0.2);
		MatrixBlock out = LibMatrixReorg.permute(in, shape, perm);

		verifyPermutation(in, out, shape, perm);
	}

	@Test
	public void testPermute3DIdentitySparse() {
		int[] shape = {5, 5, 5};
		int[] perm = {0, 1, 2};

		MatrixBlock in = generateSparseMatrixBlock(shape, 0.2);
		MatrixBlock out = LibMatrixReorg.permute(in, shape, perm);

		verifyPermutation(in, out, shape, perm);
	}

	@Test
	public void testPermute4DReverseSparse() {
		int[] shape = {2, 3, 4, 5};
		int[] perm = {3, 2, 1, 0};

		MatrixBlock in = generateSparseMatrixBlock(shape, 0.2);
		MatrixBlock out = LibMatrixReorg.permute(in, shape, perm);

		verifyPermutation(in, out, shape, perm);
	}

	@Test
	public void testPermuteHighRankSparse() {
		int[] shape = {2, 2, 2, 2, 2, 2};
		int[] perm = {5, 0, 4, 1, 3, 2};

		MatrixBlock in = generateSparseMatrixBlock(shape, 0.2);
		MatrixBlock out = LibMatrixReorg.permute(in, shape, perm);

		verifyPermutation(in, out, shape, perm);
	}

	@Test
	public void testPermute3DParallelSparse() {
		int[] shape = {100, 100, 100};
		int[] perm = {2, 0, 1};

		MatrixBlock in = generateSparseMatrixBlock(shape, 0.1);
		MatrixBlock out = LibMatrixReorg.permute(in, shape, perm, -1);

		verifyPermutation(in, out, shape, perm);
	}

	@Test
	public void testEdgeCaseEmptySparse() {
		int[] shape = {5, 5, 5};
		int[] perm = {1, 0, 2};

		MatrixBlock in = new MatrixBlock(1, 125, true);
		MatrixBlock out = LibMatrixReorg.permute(in, shape, perm);

		Assert.assertTrue(out.isEmpty());
		Assert.assertEquals(125, out.getNumColumns());
	}

	@Test
	public void testEdgeCaseSingleElementSparse() {
		int[] shape = {1, 1, 1};
		int[] perm = {2, 1, 0};

		MatrixBlock in = new MatrixBlock(1, 1, true);
		in.allocateSparseRowsBlock();
		in.appendValue(0, 0, 42.0);
		in.recomputeNonZeros();
		MatrixBlock out = LibMatrixReorg.permute(in, shape, perm);

		verifyPermutation(in, out, shape, perm);
	}

	@Test
	public void testEdgeCaseOneDimensionOneSparse() {
		int[] shape = {5, 1, 10};
		int[] perm = {2, 0, 1};

		MatrixBlock in = generateSparseMatrixBlock(shape, 0.2);
		MatrixBlock out = LibMatrixReorg.permute(in, shape, perm);

		verifyPermutation(in, out, shape, perm);
	}

	@Test
	public void testEdgeCaseTwoDimensionsOneSparse() {
		int[] shape = {1, 1, 100};
		int[] perm = {2, 1, 0};

		MatrixBlock in = generateSparseMatrixBlock(shape, 0.2);
		MatrixBlock out = LibMatrixReorg.permute(in, shape, perm);

		verifyPermutation(in, out, shape, perm);
	}

	@Test
	public void testConsecutivePermutationsSparse() {
		int[] shape = {3, 4, 5};
		int[] perm1 = {1, 0, 2};
		int[] perm2 = {2, 0, 1};

		MatrixBlock in = generateSparseMatrixBlock(shape, 0.2);
		MatrixBlock temp = LibMatrixReorg.permute(in, shape, perm1);

		int[] tempShape = {shape[perm1[0]], shape[perm1[1]], shape[perm1[2]]};
		MatrixBlock out = LibMatrixReorg.permute(temp, tempShape, perm2);

		verifyPermutation(temp, out, tempShape, perm2);
	}

	@Test
	public void testDifferentThreadCountsSparse() {
		int[] shape = {50, 50, 50};
		int[] perm = {2, 0, 1};

		MatrixBlock in = generateSparseMatrixBlock(shape, 0.1);

		MatrixBlock out1 = LibMatrixReorg.permute(in, shape, perm, 1);
		MatrixBlock out2 = LibMatrixReorg.permute(in, shape, perm, 2);
		MatrixBlock out4 = LibMatrixReorg.permute(in, shape, perm, 4);
		MatrixBlock out8 = LibMatrixReorg.permute(in, shape, perm, 8);

		long len = totalLength(shape);
		for (int i = 0; i < len; i++) {
			double v1 = out1.get(0, i);
			Assert.assertEquals(v1, out2.get(0, i), 0.0001);
			Assert.assertEquals(v1, out4.get(0, i), 0.0001);
			Assert.assertEquals(v1, out8.get(0, i), 0.0001);
		}
	}

	@Test
	public void testThreadCountExceedsElementCountSparse() {
		int[] shape = {2, 2};
		int[] perm = {1, 0};
		MatrixBlock in = generateSparseMatrixBlock(shape, 0.5);

		MatrixBlock out1 = LibMatrixReorg.permute(in, shape, perm, 1);
		MatrixBlock out16 = LibMatrixReorg.permute(in, shape, perm, 16);

		verifyPermutation(in, out16, shape, perm);
		long len = totalLength(shape);
		for (int i = 0; i < len; i++)
			Assert.assertEquals(out1.get(0, i), out16.get(0, i), 0.0001);
	}

	@Test
	public void testPermuteAllDimensionsCyclicSparse() {
		int[] shape = {3, 4, 5, 2};
		int[] perm = {1, 2, 3, 0};

		MatrixBlock in = generateSparseMatrixBlock(shape, 0.2);
		MatrixBlock out = LibMatrixReorg.permute(in, shape, perm);

		verifyPermutation(in, out, shape, perm);
	}

	@Test
	public void testPermuteNonContiguousStridesSparse() {
		int[] shape = {7, 11, 13};
		int[] perm = {2, 0, 1};

		MatrixBlock in = generateSparseMatrixBlock(shape, 0.2);
		MatrixBlock out = LibMatrixReorg.permute(in, shape, perm);

		verifyPermutation(in, out, shape, perm);
	}

	@Test
	public void testPermuteLargePrimeStridesSparse() {
		int[] shape = {17, 19};
		int[] perm = {1, 0};

		MatrixBlock in = generateSparseMatrixBlock(shape, 0.2);
		MatrixBlock out = LibMatrixReorg.permute(in, shape, perm);

		verifyPermutation(in, out, shape, perm);
	}

	@Test
	public void testSpecialFloatingPointValuesSparse() {
		int[] shape = {2, 2};
		int[] perm = {1, 0};

		MatrixBlock in = new MatrixBlock(1, 4, true);
		in.allocateSparseRowsBlock();
		in.appendValue(0, 0, Double.NaN);
		in.appendValue(0, 1, Double.POSITIVE_INFINITY);
		in.appendValue(0, 2, Double.NEGATIVE_INFINITY);
		in.recomputeNonZeros();

		MatrixBlock out = LibMatrixReorg.permute(in, shape, perm);

		int nanCount = 0, posInfCount = 0, negInfCount = 0;
		for (int i = 0; i < 4; i++) {
			double v = out.get(0, i);
			if (Double.isNaN(v)) nanCount++;
			else if (v == Double.POSITIVE_INFINITY) posInfCount++;
			else if (v == Double.NEGATIVE_INFINITY) negInfCount++;
		}
		Assert.assertEquals(1, nanCount);
		Assert.assertEquals(1, posInfCount);
		Assert.assertEquals(1, negInfCount);
	}

	@Test
	public void testInputImmutableAfterPermuteSparse() {
		int[] shape = {4, 5, 6};
		int[] perm = {2, 0, 1};

		MatrixBlock in = generateSparseMatrixBlock(shape, 0.2);
		long len = totalLength(shape);
		double[] before = new double[(int) len];
		for (int i = 0; i < len; i++)
			before[i] = in.get(0, i);

		LibMatrixReorg.permute(in, shape, perm);

		for (int i = 0; i < len; i++)
			Assert.assertEquals(before[i], in.get(0, i), 0.0001);
	}

	@Test
	public void testLargeSparseManyRowsConsistency() {
		int[] shape = {10, 10, 10};
		int[] perm = {2, 0, 1};
		MatrixBlock in = generateSparseMatrixBlock(shape, 0.3);

		MatrixBlock outSingle = LibMatrixReorg.permute(in, shape, perm, 1);
		MatrixBlock outMulti = LibMatrixReorg.permute(in, shape, perm, -1);

		verifyPermutation(in, outSingle, shape, perm);
		long len = totalLength(shape);
		for (int i = 0; i < len; i++)
			Assert.assertEquals(outSingle.get(0, i), outMulti.get(0, i), 0.0001);
	}

	@Test
	public void testNonZeroCountPreservedAfterPermuteSparse() {
		int[] shape = {6, 7, 8};
		int[] perm = {2, 0, 1};
		MatrixBlock in = generateSparseMatrixBlock(shape, 0.25);
		long nnzBefore = in.getNonZeros();

		MatrixBlock out = LibMatrixReorg.permute(in, shape, perm);
		out.recomputeNonZeros();

		Assert.assertEquals(nnzBefore, out.getNonZeros());
		Assert.assertEquals(nnzBefore == 0, out.isEmpty());
	}

	@Test
	public void testDenseSparseParitySmall() {
		int[] shape = {4, 3, 5};
		int[] perm = {2, 0, 1};

		MatrixBlock dense = generateDenseMatrixBlock(shape);
		MatrixBlock sparse = denseToSparse(dense);

		MatrixBlock outDense = LibMatrixReorg.permute(dense, shape, perm);
		MatrixBlock outSparse = LibMatrixReorg.permute(sparse, shape, perm);

		long len = totalLength(shape);
		for (int i = 0; i < len; i++) {
			Assert.assertEquals(outDense.get(0, i), outSparse.get(0, i), 0.0001);
		}
	}

	@Test
	public void testDenseSparseParityHighRank() {
		int[] shape = {2, 3, 4, 5};
		int[] perm = {3, 1, 0, 2};

		MatrixBlock dense = generateDenseMatrixBlock(shape);
		MatrixBlock sparse = denseToSparse(dense);

		MatrixBlock outDense = LibMatrixReorg.permute(dense, shape, perm);
		MatrixBlock outSparse = LibMatrixReorg.permute(sparse, shape, perm);

		long len = totalLength(shape);
		for (int i = 0; i < len; i++) {
			Assert.assertEquals(outDense.get(0, i), outSparse.get(0, i), 0.0001);
		}
	}

	@Test
	public void testInvalidPermutationDuplicateIndicesDense() {
		int[] shape = {2, 3, 4};
		int[] perm = {0, 0, 2};

		MatrixBlock in = generateDenseMatrixBlock(shape);
		try {
			LibMatrixReorg.permute(in, shape, perm);
			Assert.fail("Expected an exception for duplicate indices");
		} catch (Exception expected) {
		}
	}

	@Test
	public void testInvalidPermutationOutOfRangeIndexDense() {
		int[] shape = {2, 3, 4};
		int[] perm = {0, 1, 5};
		MatrixBlock in = generateDenseMatrixBlock(shape);
		try {
			LibMatrixReorg.permute(in, shape, perm);
			Assert.fail("Expected an exception for an out-of-range permutation index");
		} catch (Exception expected) {
		}
	}

	@Test
	public void testInvalidPermutationWrongLengthDense() {
		int[] shape = {2, 3, 4};
		int[] perm = {1, 0};
		MatrixBlock in = generateDenseMatrixBlock(shape);
		try {
			LibMatrixReorg.permute(in, shape, perm);
			Assert.fail("Expected an exception for a permutation array of the wrong length");
		} catch (Exception expected) {
		}
	}

	@Test
	public void testInvalidShapeDataLengthMismatchDense() {
		int[] shape = {3, 3, 3};
		int[] perm = {1, 0, 2};
		MatrixBlock in = generateDenseMatrixBlock(new int[]{2, 2, 2});
		try {
			LibMatrixReorg.permute(in, shape, perm);
			Assert.fail("Expected an exception when shape does not match input data length");
		} catch (Exception expected) {
		}
	}

	@Test
	public void testInvalidPermutationDuplicateIndicesSparse() {
		int[] shape = {2, 3, 4};
		int[] perm = {0, 0, 2};
		MatrixBlock in = generateSparseMatrixBlock(shape, 0.2);
		try {
			LibMatrixReorg.permute(in, shape, perm);
			Assert.fail("Expected an exception for duplicate indices");
		} catch (Exception expected) {
		}
	}

	@Test
	public void testSparseFormatPreservedAfterPermute() {
		int[] shape = {6, 7, 8};
		int[] perm = {2, 0, 1};

		MatrixBlock in = generateSparseMatrixBlock(shape, 0.05);
		MatrixBlock out = LibMatrixReorg.permute(in, shape, perm);

		Assert.assertTrue("Sparse input must produce a sparse output", out.isInSparseFormat());
	}

	@Test
	public void testDenseFormatPreservedAfterPermute() {
		int[] shape = {6, 7, 8};
		int[] perm = {2, 0, 1};

		MatrixBlock in = generateDenseMatrixBlock(shape);
		MatrixBlock out = LibMatrixReorg.permute(in, shape, perm);

		Assert.assertFalse("Dense input must produce a dense output", out.isInSparseFormat());
	}

	/*
	This Test takes relatively long. 
	The smaller sparse tests are relatively unstable, sometimes showing some speedup, sometimes not. 
	The larg test however, always shows some speedup but in the range between 1.01 and 10.4. 
	*/
	@Test 
	@Ignore
	public void testPerformanceBenchmarkDenseAndSparse() {
		int size = 100;
		int[] shape = {size, size, size};
		int[] perm = {2, 0, 1};

		System.out.println("--- Benchmark Results (Size " + size + "^3) ---");

		MatrixBlock denseIn = generateDenseMatrixBlock(shape);
		double dSingle = benchmarkPermute(denseIn, shape, perm, 1);
		double dMulti = benchmarkPermute(denseIn, shape, perm, -1);
		printBenchmarkResult("Dense", dSingle, dMulti);

		MatrixBlock sparseIn10 = generateSparseMatrixBlock(shape, 0.1);
		double s10Single = benchmarkPermute(sparseIn10, shape, perm, 1);
		double s10Multi = benchmarkPermute(sparseIn10, shape, perm, -1);
		printBenchmarkResult("Sparse (0.1)", s10Single, s10Multi);

		MatrixBlock sparseIn50 = generateSparseMatrixBlock(shape, 0.5);
		double s50Single = benchmarkPermute(sparseIn50, shape, perm, 1);
		double s50Multi = benchmarkPermute(sparseIn50, shape, perm, -1);
		printBenchmarkResult("Sparse (0.5)", s50Single, s50Multi);

		MatrixBlock sparseLarge = generateSparseMatrixBlock(new int[]{100, 1000, 1000}, 0.5);
		double sLargeSingle = benchmarkPermute(sparseLarge, new int[]{100, 1000, 1000}, new int[]{2, 0, 1}, 1);
		double sLargeMulti = benchmarkPermute(sparseLarge, new int[]{100, 1000, 1000}, new int[]{2, 0, 1}, -1);
		printBenchmarkResult("Sparse Large (0.5)", sLargeSingle, sLargeMulti);

		Assert.assertTrue(dMulti < dSingle * 2.0);
	}

	@Test
	@Ignore // Unstable on my device 
	public void testPerformanceSingleVsMultiThreadedDense() {
		int size = 100;
		int[] shape = {size, size, size};
		int[] perm = {2, 0, 1};

		MatrixBlock in = generateDenseMatrixBlock(shape);

		MatrixBlock outSingle = LibMatrixReorg.permute(in, shape, perm, 1);
		MatrixBlock outMulti = LibMatrixReorg.permute(in, shape, perm, -1);
		verifyPermutation(in, outSingle, shape, perm);
		verifyPermutation(in, outMulti, shape, perm);

		double timeSingle = benchmarkPermute(in, shape, perm, 1);
		double timeMulti = benchmarkPermute(in, shape, perm, -1);

		System.out.println("Large Matrix (" + size + "x" + size + "x" + size + "):");
		System.out.printf("Single-threaded (avg): %.2f ms%n", timeSingle);
		System.out.printf("Multi-threaded (avg): %.2f ms%n", timeMulti);
		System.out.printf("Speedup: %.2fx%n", timeSingle / timeMulti);

		Assert.assertTrue("Multi-threaded should be faster for large matrices", timeMulti < timeSingle);
	}

	@Test // Shows a stable speedup of ~2,20 
	@Ignore
	public void testPerformanceLargeMatrixSingleVsMultiDense() {
		int[] shape = {10, 1000, 1000};
		int[] perm = {0, 2, 1};

		MatrixBlock in = generateDenseMatrixBlock(shape);

		double timeSingle = benchmarkPermute(in, shape, perm, 1);
		double timeMulti = benchmarkPermute(in, shape, perm, -1);

		System.out.println("Large Matrix (10x1000x1000):");
		System.out.printf("Single-threaded (avg): %.2f ms%n", timeSingle);
		System.out.printf("Multi-threaded (avg): %.2f ms%n", timeMulti);
		System.out.printf("Speedup: %.2fx%n", timeSingle / timeMulti);

		Assert.assertTrue("Multi-threaded should be faster for large matrices", timeMulti < timeSingle);
	}

	private MatrixBlock generateDenseMatrixBlock(int[] shape) {
		long len = 1;
		for (int d : shape) len *= d;

		int rows = shape[0];
		int cols = (int) (len / rows);

		MatrixBlock mb = new MatrixBlock(rows, cols, false);
		mb.allocateDenseBlock();
		
		double[] data = mb.getDenseBlockValues();
		for (int i = 0; i < data.length; i++) {
			data[i] = (double) i;
		}
		return mb;
	}

	private MatrixBlock generateSparseMatrixBlock(int[] shape, double sparsity) {
		long len = 1;
		for (int d : shape) len *= d;

		int rows = shape[0];
		int cols = (int) (len / rows);

		MatrixBlock mb = new MatrixBlock(rows, cols, true);
		mb.allocateSparseRowsBlock();
		
		for (long i = 0; i < len; i++) {
			if ((i * 0.12345) % 1.0 < sparsity) {
				int r = (int) (i / cols);
				int c = (int) (i % cols);
				mb.appendValue(r, c, (double) i);
			}
		}
		mb.recomputeNonZeros();
		return mb;
	}

	private MatrixBlock denseToSparse(MatrixBlock dense) { 
		int rows = dense.getNumRows();
		int cols = dense.getNumColumns();

		MatrixBlock sparse = new MatrixBlock(rows, cols, true);
		sparse.allocateSparseRowsBlock();
		
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				double val = dense.get(r, c);
				if (val != 0) {
					sparse.appendValue(r, c, val);
				}
			}
		}
		sparse.recomputeNonZeros();
		return sparse;
	}

	private void verifyPermutation(MatrixBlock in, MatrixBlock out, int[] inShape, int[] perm) {
		int rank = inShape.length;
		int[] outShape = new int[rank];
		for (int i = 0; i < rank; i++)
			outShape[i] = inShape[perm[i]];

		long[] outStrides = getStrides(outShape);
		long[] inStrides = getStrides(inShape);
		
		long len = 1;
		for (int d : outShape) len *= d;

		for (long i = 0; i < len; i++) {
			int[] outCoords = new int[rank];
			long remaining = i;
			for (int d = 0; d < rank; d++) {
				outCoords[d] = (int) (remaining / outStrides[d]);
				remaining %= outStrides[d];
			}

			int[] inCoords = new int[rank];
			for (int d = 0; d < rank; d++){
				inCoords[perm[d]] = outCoords[d];
			}

			long inIndex = 0;
			for (int d = 0; d < rank; d++){
				inIndex += inCoords[d] * inStrides[d];
			}

			int inCols = in.getNumColumns();
			int r = (int) (inIndex / inCols);
			int c = (int) (inIndex % inCols);
			
			double expectedValue = in.get(r, c);
			double actualValue = out.get(0, (int)i);

			if (Math.abs(expectedValue - actualValue) > 0.0001) {
				Assert.fail("Mismatch at linear output index " + i + 
							". Output coords " + Arrays.toString(outCoords) + 
							". Input coords " + Arrays.toString(inCoords) +
							". Expected " + expectedValue + " but got " + actualValue);
			}
		}
	}

	private long[] getStrides(int[] dims) {
		long[] strides = new long[dims.length];
		long stride = 1;
		for (int i = dims.length - 1; i >= 0; i--) {
			strides[i] = stride;
			stride *= dims[i];
		}
		return strides;
	}

	private long totalLength(int[] shape) {
		long len = 1;
		for (int d : shape) len *= d;
		return len;
	}

	private double benchmarkPermute(MatrixBlock in, int[] shape, int[] perm, int k) {
		double checksum = 0;
		for (int i = 0; i < 5; i++)
			checksum += LibMatrixReorg.permute(in, shape, perm, k).get(0, 0);

		long start = System.nanoTime();
		for (int i = 0; i < 10; i++)
			checksum += LibMatrixReorg.permute(in, shape, perm, k).get(0, 0);
		long elapsed = System.nanoTime() - start;

		if (checksum == Double.NaN)
			System.out.println("unreachable");
		return (double) elapsed / 1_000_000.0 / 10.0;
	}

	private void printBenchmarkResult(String label, double single, double multi) {
		System.out.printf("%-20s Single: %6.2fms  Multi: %6.2fms  Speedup: %.2fx%n",
			label, single, multi, single / multi);
	}
}