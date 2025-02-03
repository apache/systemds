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
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.util.HashMap;
import java.util.Map;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.CompressedMatrixBlockFactory;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.CompressionSettingsBuilder;
import org.apache.sysds.runtime.compress.CompressionStatistics;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.cost.ACostEstimate;
import org.apache.sysds.runtime.compress.cost.CostEstimatorBuilder;
import org.apache.sysds.runtime.compress.cost.CostEstimatorFactory;
import org.apache.sysds.runtime.compress.cost.InstructionTypeCounter;
import org.apache.sysds.runtime.compress.lib.CLALibCBind;
import org.apache.sysds.runtime.compress.lib.CLALibReplace;
import org.apache.sysds.runtime.compress.workload.WTreeRoot;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.test.component.compress.workload.WorkloadTest;
import org.junit.Test;

public class CompressedCustomTests {
	protected static final Log LOG = LogFactory.getLog(CompressedCustomTests.class.getName());

	@Test
	public void compressNaNDense() {
		try {
			MatrixBlock m = new MatrixBlock(100, 100, Double.NaN);

			MatrixBlock m2 = CompressedMatrixBlockFactory.compress(m).getLeft();

			for(int i = 0; i < m.getNumRows(); i++)
				for(int j = 0; j < m.getNumColumns(); j++)
					assertEquals(Double.NaN, m2.get(i, j), 0.0);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void compressNaNSparse() {
		try {

			MatrixBlock m = new MatrixBlock(100, 100, true);
			for(int i = 0; i < m.getNumRows(); i++)
				m.set(i, i, Double.NaN);
			assertTrue(m.isInSparseFormat());
			MatrixBlock m2 = CompressedMatrixBlockFactory.compress(m).getLeft();

			for(int i = 0; i < m.getNumRows(); i++)
				for(int j = 0; j < m.getNumColumns(); j++) {
					if(i == j)
						assertEquals(Double.NaN, m2.get(i, j), 0.0);
					else
						assertEquals(0.0, m2.get(i, j), 0.0);
				}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void workloadTreeInterface() {
		try {
			MatrixBlock m = TestUtils.generateTestMatrixBlock(100, 4, 1, 1, 0.5, 231);
			Map<String, String> args = new HashMap<>();
			args.put("$1", "src/test/resources/component/compress/1-1.csv");
			WTreeRoot wtr = WorkloadTest.getWorkloadTree("functions/scale.dml", args);
			MatrixBlock m2 = CompressedMatrixBlockFactory.compress(m, 10, wtr).getLeft();
			TestUtils.compareMatricesBitAvgDistance(m, m2, 0, 0);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void workloadTreeInterface2() {
		try {
			MatrixBlock m = TestUtils.generateTestMatrixBlock(100, 4, 1, 1, 0.5, 54);
			Map<String, String> args = new HashMap<>();
			args.put("$1", "src/test/resources/component/compress/1-1.csv");
			WTreeRoot wtr = WorkloadTest.getWorkloadTree("functions/scale.dml", args);
			MatrixBlock m2 = CompressedMatrixBlockFactory.compress(m, wtr).getLeft();
			TestUtils.compareMatricesBitAvgDistance(m, m2, 0, 0);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void costEstimatorBuilder() {
		try {
			MatrixBlock m = TestUtils.generateTestMatrixBlock(100, 4, 1, 1, 0.5, 1612323);
			Map<String, String> args = new HashMap<>();
			args.put("$1", "src/test/resources/component/compress/1-1.csv");
			WTreeRoot wtr = WorkloadTest.getWorkloadTree("functions/scale.dml", args);
			CostEstimatorBuilder csb = new CostEstimatorBuilder(wtr);
			MatrixBlock m2 = CompressedMatrixBlockFactory.compress(m, csb).getLeft();
			TestUtils.compareMatricesBitAvgDistance(m, m2, 0, 0);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void costEstimatorBuilder2() {
		try {
			MatrixBlock m = TestUtils.generateTestMatrixBlock(100, 4, 1, 1, 0.5, 4442);
			Map<String, String> args = new HashMap<>();
			args.put("$1", "src/test/resources/component/compress/1-1.csv");
			WTreeRoot wtr = WorkloadTest.getWorkloadTree("functions/scale.dml", args);
			CostEstimatorBuilder csb = new CostEstimatorBuilder(wtr);
			MatrixBlock m2 = CompressedMatrixBlockFactory.compress(m, 10, csb).getLeft();
			TestUtils.compareMatricesBitAvgDistance(m, m2, 0, 0);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void instructionTypeCounter() {
		try {
			MatrixBlock m = TestUtils.generateTestMatrixBlock(100, 4, 1, 1, 0.5, 4442);
			InstructionTypeCounter ins = new InstructionTypeCounter(0, 0, 0, 0, 24, 15, 0, 0, false);
			MatrixBlock m2 = CompressedMatrixBlockFactory.compress(m, 10, ins).getLeft();
			TestUtils.compareMatricesBitAvgDistance(m, m2, 0, 0);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void instructionTypeCounter2() {
		try {
			MatrixBlock m = TestUtils.generateTestMatrixBlock(100, 4, 1, 1, 0.5, 521);
			InstructionTypeCounter ins = new InstructionTypeCounter(0, 0, 0, 0, 24, 15, 0, 0, false);
			MatrixBlock m2 = CompressedMatrixBlockFactory.compress(m, ins).getLeft();
			TestUtils.compareMatricesBitAvgDistance(m, m2, 0, 0);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void instructionTypeCounterNull() {
		try {
			MatrixBlock m = TestUtils.generateTestMatrixBlock(100, 4, 1, 1, 0.5, 521);
			InstructionTypeCounter ins = null;
			MatrixBlock m2 = CompressedMatrixBlockFactory.compress(m, ins).getLeft();
			TestUtils.compareMatricesBitAvgDistance(m, m2, 0, 0);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void instructionTypeCounterNull2() {
		try {
			MatrixBlock m = TestUtils.generateTestMatrixBlock(100, 4, 1, 1, 0.5, 521);
			InstructionTypeCounter ins = null;
			MatrixBlock m2 = CompressedMatrixBlockFactory.compress(m, 14, ins).getLeft();
			TestUtils.compareMatricesBitAvgDistance(m, m2, 0, 0);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void builder() {
		try {
			MatrixBlock m = TestUtils.generateTestMatrixBlock(100, 4, 1, 1, 0.5, 521);
			CompressionSettingsBuilder sb = new CompressionSettingsBuilder();

			MatrixBlock m2 = CompressedMatrixBlockFactory.compress(m, sb).getLeft();
			TestUtils.compareMatricesBitAvgDistance(m, m2, 0, 0);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void builder2() {
		try {
			MatrixBlock m = TestUtils.generateTestMatrixBlock(100, 4, 1, 1, 0.5, 1313131);
			CompressionSettingsBuilder sb = new CompressionSettingsBuilder();
			MatrixBlock m2 = CompressedMatrixBlockFactory.compress(m, 16, sb).getLeft();
			TestUtils.compareMatricesBitAvgDistance(m, m2, 0, 0);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void normal() {
		try {
			MatrixBlock m = TestUtils.generateTestMatrixBlock(100, 4, 1, 1, 0.5, 42145);
			MatrixBlock m2 = CompressedMatrixBlockFactory.compress(m).getLeft();
			TestUtils.compareMatricesBitAvgDistance(m, m2, 0, 0);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void threaded() {
		try {
			MatrixBlock m = TestUtils.generateTestMatrixBlock(100, 4, 1, 1, 0.5, 42145);
			MatrixBlock m2 = CompressedMatrixBlockFactory.compress(m, 9).getLeft();
			TestUtils.compareMatricesBitAvgDistance(m, m2, 0, 0);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void costEstimator() {
		try {
			MatrixBlock m = TestUtils.generateTestMatrixBlock(100, 4, 1, 1, 0.5, 4442);
			Map<String, String> args = new HashMap<>();
			args.put("$1", "src/test/resources/component/compress/1-1.csv");
			WTreeRoot wtr = WorkloadTest.getWorkloadTree("functions/scale.dml", args);
			CostEstimatorBuilder csb = new CostEstimatorBuilder(wtr);
			CompressionSettings cs = new CompressionSettingsBuilder().create();
			ACostEstimate ce = CostEstimatorFactory.create(cs, csb, m.getNumRows(), m.getNumColumns(), m.getSparsity());
			MatrixBlock m2 = CompressedMatrixBlockFactory.compress(m, 10, ce).getLeft();
			TestUtils.compareMatricesBitAvgDistance(m, m2, 0, 0);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void costEstimator2() {
		try {
			MatrixBlock m = TestUtils.generateTestMatrixBlock(100, 4, 1, 1, 0.5, 4442);
			Map<String, String> args = new HashMap<>();
			args.put("$1", "src/test/resources/component/compress/1-1.csv");
			WTreeRoot wtr = WorkloadTest.getWorkloadTree("functions/scale.dml", args);
			CostEstimatorBuilder csb = new CostEstimatorBuilder(wtr);
			CompressionSettings cs = new CompressionSettingsBuilder().create();
			ACostEstimate ce = CostEstimatorFactory.create(cs, csb, m.getNumRows(), m.getNumColumns(), m.getSparsity());
			MatrixBlock m2 = CompressedMatrixBlockFactory.compress(m, ce).getLeft();
			TestUtils.compareMatricesBitAvgDistance(m, m2, 0, 0);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test(expected = DMLCompressionException.class)
	public void negativeCreateConstant() {
		CompressedMatrixBlockFactory.createConstant(-1, 1, 3241);

	}

	@Test(expected = DMLCompressionException.class)
	public void negativeCreateConstant2() {
		CompressedMatrixBlockFactory.createConstant(32, -1, 3241);
	}

	@Test(expected = DMLCompressionException.class)
	public void negativeCreateConstant3() {
		CompressedMatrixBlockFactory.createConstant(32, 0, 3241);
	}

	@Test(expected = DMLCompressionException.class)
	public void negativeCreateConstant4() {
		CompressedMatrixBlockFactory.createConstant(0, 321, 3241);
	}

	@Test(expected = DMLCompressionException.class)
	public void negativeCreateConstant5() {
		CompressedMatrixBlockFactory.createConstant(-1, -1, 3241);
	}

	@Test
	public void createConstant() {
		MatrixBlock mb = CompressedMatrixBlockFactory.createConstant(10, 10, 3241);
		MatrixBlock mb2 = new MatrixBlock(10, 10, 3241.0);
		TestUtils.compareMatricesBitAvgDistance(mb, mb2, 0, 0);
	}

	@Test
	public void createUncompressedCompressedMatrixBlockTest() {
		MatrixBlock mb = TestUtils.generateTestMatrixBlock(32, 42, 32, 123, 0.2, 2135);
		MatrixBlock mb2 = CompressedMatrixBlockFactory.genUncompressedCompressedMatrixBlock(mb);
		TestUtils.compareMatricesBitAvgDistance(mb, mb2, 0, 0);
	}

	@Test
	public void notInvalidIfNnzNotSet() {
		MatrixBlock mb = TestUtils.generateTestMatrixBlock(32, 42, 32, 123, 0.2, 2135);
		mb.setNonZeros(-23L);
		CompressedMatrixBlockFactory.compress(mb);
	}

	@Test
	public void statisticsStartInfinite() {
		CompressionStatistics cs = new CompressionStatistics();
		String s = cs.toString();
		assertTrue(s.contains("Infinity"));
	}

	@Test(expected = DMLCompressionException.class)
	public void setTransposeIncorrect() {
		new CompressionSettingsBuilder().setTransposeInput("bb");
	}

	@Test
	public void compressSingleCol() {
		MatrixBlock m1 = TestUtils.generateTestMatrixBlock(1000, 1, 1, 1, 0.8, 231);
		MatrixBlock m2 = CompressedMatrixBlockFactory.compress(m1).getLeft();
		TestUtils.compareMatrices(m1, m2, 0, "no");
	}

	@Test
	public void manyColsSparse() {
		MatrixBlock m0 = new MatrixBlock(1000, 10000, 0.0);
		MatrixBlock m1 = TestUtils.generateTestMatrixBlock(1000, 1, 1, 1, 0.01, 231);
		m1 = m0.append(m1);
		MatrixBlock m2 = CompressedMatrixBlockFactory.compress(m1).getLeft();

		TestUtils.compareMatricesBitAvgDistance(m1, m2, 0, 0, "no");
	}

	@Test
	public void manyRowsSparse() {
		MatrixBlock m0 = new MatrixBlock(500001, 10, 0.0);
		MatrixBlock m1 = TestUtils.generateTestMatrixBlock(500001, 1, 1, 1, 0.003, 231);
		m1 = m0.append(m1);
		MatrixBlock m2 = CompressedMatrixBlockFactory.compress(m1).getLeft();
		TestUtils.compareMatricesBitAvgDistance(m1, m2, 0, 0, "no");
	}

	@Test
	public void manyRowsButNotQuite() {
		MatrixBlock m0 = new MatrixBlock(10001, 10, 0.0);
		MatrixBlock m1 = TestUtils.generateTestMatrixBlock(10001, 1, 1, 1, 0.11, 231);
		m1 = m0.append(m1);
		MatrixBlock m2 = CompressedMatrixBlockFactory.compress(m1).getLeft();
		TestUtils.compareMatricesBitAvgDistance(m1, m2, 0, 0, "no");
	}

	@Test(expected = Exception.class)
	public void cbindWithError() {
		CLALibCBind.cbind(null, new MatrixBlock[] {null}, 0);
	}

	@Test(expected = Exception.class)
	public void replaceWithError() {
		CLALibReplace.replace(null, null, 0, 0, 10);
	}

	@Test
	public void replaceInf() {
		assertNull(CLALibReplace.replace(null, null, Double.POSITIVE_INFINITY, 0, 10));
	}
}
