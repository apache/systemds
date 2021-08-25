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

package org.apache.sysds.test.component.compress;

import static org.junit.Assert.assertEquals;

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.ArrayList;

import org.apache.commons.lang.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressedMatrixBlockFactory;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.CompressionSettingsBuilder;
import org.apache.sysds.runtime.compress.cocode.CoCoderFactory;
import org.apache.sysds.runtime.compress.colgroup.AColGroup.CompressionType;
import org.apache.sysds.runtime.compress.cost.CostEstimatorFactory;
import org.apache.sysds.runtime.compress.estim.sample.SampleEstimatorFactory.EstimationType;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public class CompressedSingleTests {

	protected static final Log LOG = LogFactory.getLog(CompressedSingleTests.class.getName());
	protected static final int rows = 1000;
	protected static final int cols = 2;
	protected static final double min = 0;
	protected static final double max = 100;
	protected static final double sparsity = 0.6;
	protected static final int seed = 2;

	protected final MatrixBlock mb;
	protected final CompressedMatrixBlock cmb;

	public CompressedSingleTests() {
		double[][] input = TestUtils.round(TestUtils.generateTestMatrix(rows, cols, min, max, sparsity, seed));
		mb = DataConverter.convertToMatrixBlock(input);
		mb.recomputeNonZeros();
		mb.examSparsity();

		cmb = (CompressedMatrixBlock) CompressedMatrixBlockFactory.compress(mb).getLeft();
	}

	@Test
	public void binaryOperationsInPlace() {
		MatrixBlock refmb = new MatrixBlock();
		refmb.copy(mb, true);
		MatrixValue thatValue = DataConverter
			.convertToMatrixBlock(TestUtils.generateTestMatrix(rows, cols, min, max, sparsity, seed + 13));
		BinaryOperator op = new BinaryOperator(Multiply.getMultiplyFnObject());

		// reference
		MatrixBlock ret1 = refmb.binaryOperationsInPlace(op, thatValue);
		// compressed
		MatrixBlock ret2 = cmb.binaryOperationsInPlace(op, thatValue);

		double[][] d1 = DataConverter.convertToDoubleMatrix(ret1);
		double[][] d2 = DataConverter.convertToDoubleMatrix(ret2);

		TestUtils.compareMatricesBitAvgDistance(d1, d2, 1024, 1024, "Binary Operations Inplace Failed.");
	}

	@Test(expected = DMLRuntimeException.class)
	public void invalid_call_append_listOfMatrixBlock() {
		cmb.append(new MatrixBlock[] {new MatrixBlock(10, 10, 3)}, null, false);
	}

	@Test(expected = NullPointerException.class)
	public void invalid_call_appendArray() {
		cmb.append(null, new ArrayList<IndexedMatrixValue>(), 1, false, false, 1);
	}

	@Test
	public void estimateIsEqualToActualSizeInMemory() {
		assertEquals(cmb.getInMemorySize(), cmb.estimateSizeInMemory());
	}

	@Test
	public void estimateIsEqualToActualSizeDisk() {
		assertEquals(cmb.getExactSizeOnDisk(), cmb.estimateSizeOnDisk());
	}

	@Test
	public void testOnDiskSizeInBytes() {
		try {
			// Serialize out
			ByteArrayOutputStream bos = new ByteArrayOutputStream();
			DataOutputStream fos = new DataOutputStream(bos);
			cmb.write(fos);
			byte[] arr = bos.toByteArray();
			int size = arr.length;
			assertEquals(cmb.getClass().getSimpleName() + "\n" + cmb.toString() + "\n", size, cmb.getExactSizeOnDisk());
		}
		catch(IOException e) {
			throw new RuntimeException("Error in io", e);
		}
		catch(Exception e) {
			e.printStackTrace();
			throw e;
		}
	}

	@Test(expected = NotImplementedException.class)
	public void test_copyShallow() {
		CompressedMatrixBlock copyIntoMe = new CompressedMatrixBlock();
		copyIntoMe.copyShallow(cmb);
	}

	@Test
	public void test_settingsBuilder() {
		CompressionSettingsBuilder b = new CompressionSettingsBuilder();
		b = b.addValidCompression(CompressionType.CONST).setLossy(true).setLossy(false).setSortValuesByLength(true)
			.setAllowSharedDictionary(true).setColumnPartitioner(CoCoderFactory.PartitionerType.BIN_PACKING)
			.setMaxColGroupCoCode(3).setEstimationType(EstimationType.ShlosserJackknifeEstimator)
			.clearValidCompression().setSamplingRatio(0.2).setSeed(1342).setCoCodePercentage(0.22)
			.setMinimumSampleSize(1342).setCostType(CostEstimatorFactory.CostType.MEMORY);
		CompressionSettings s = b.create();
		b = b.copySettings(s);
	}
}
