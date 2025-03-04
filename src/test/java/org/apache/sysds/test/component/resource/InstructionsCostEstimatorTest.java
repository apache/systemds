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

package org.apache.sysds.test.component.resource;

import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.resource.CloudInstance;
import org.apache.sysds.resource.ResourceCompiler;
import org.apache.sysds.resource.cost.CostEstimationException;
import org.apache.sysds.resource.cost.CostEstimator;
import org.apache.sysds.resource.cost.RDDStats;
import org.apache.sysds.resource.cost.VarStats;
import org.apache.sysds.runtime.controlprogram.Program;
import org.apache.sysds.runtime.instructions.cp.BinaryCPInstruction;
import org.apache.sysds.runtime.instructions.cp.CPInstruction;
import org.apache.sysds.runtime.instructions.cp.VariableCPInstruction;
import org.apache.sysds.runtime.instructions.spark.BinarySPInstruction;
import org.apache.sysds.runtime.instructions.spark.SPInstruction;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.util.HashMap;

import static org.apache.sysds.resource.CloudUtils.GBtoBytes;
import static org.apache.sysds.test.component.resource.ResourceTestUtils.getSimpleCloudInstanceMap;

public class InstructionsCostEstimatorTest {
	private static final HashMap<String, CloudInstance> instanceMap = getSimpleCloudInstanceMap();

	private CostEstimator estimator;

	@Before
	public void setup() {
		ResourceCompiler.setSparkClusterResourceConfigs(GBtoBytes(8), 4, 4, GBtoBytes(8), 4);
		estimator = new CostEstimator(new Program(), instanceMap.get("m5.xlarge"), instanceMap.get("m5.xlarge"));
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Tests for CP Instructions
	
	@Test
	public void createvarMatrixVariableCPInstructionTest() throws CostEstimationException {
		String instDefinition = "CP°createvar°testVar°testOutputFile°false°MATRIX°binary°100°100°1000°10000°COPY";
		VariableCPInstruction inst = VariableCPInstruction.parseInstruction(instDefinition);
		testGettingTimeEstimateForCPInst(estimator, null, inst, 0);
		// test the proper maintainCPInstVariableStatistics functionality
		estimator.maintainStats(inst);
		VarStats actualStats = estimator.getStats("testVar");
		Assert.assertNotNull(actualStats);
		Assert.assertEquals(10000, actualStats.getCells());
	}

	@Test
	public void createvarFrameVariableCPInstructionTest() throws CostEstimationException {
		String instDefinition = "CP°createvar°testVar°testOutputFile°false°FRAME°binary°100°100°1000°10000°COPY";
		VariableCPInstruction inst = VariableCPInstruction.parseInstruction(instDefinition);
		testGettingTimeEstimateForCPInst(estimator, null, inst, 0);
		// test the proper maintainCPInstVariableStatistics functionality
		estimator.maintainStats(inst);
		VarStats actualStats = estimator.getStats("testVar");
		Assert.assertNotNull(actualStats);
		Assert.assertEquals(10000, actualStats.getCells());
	}

	@Test
	public void createvarInvalidVariableCPInstructionTest() throws CostEstimationException {
		String instDefinition = "CP°createvar°testVar°testOutputFile°false°TENSOR°binary°100°100°1000°10000°copy";
		VariableCPInstruction inst = VariableCPInstruction.parseInstruction(instDefinition);
		try {
			estimator.maintainStats(inst);
			testGettingTimeEstimateForCPInst(estimator, null, inst, 0);
			Assert.fail("Tensor is not supported by the cost estimator");
		} catch (RuntimeException e) {
			// needed catch block to assert that RuntimeException has been thrown
		}
	}

	@Test
	public void randCPInstructionTest() throws CostEstimationException {
		HashMap<String, VarStats> inputStats = new HashMap<>();
		inputStats.put("matrixVar", generateStats("matrixVar", 10000, 10000, -1));
		inputStats.put("outputVar", generateStats("outputVar", 10000, 10000, -1));

		String instDefinition = "CP°+°scalarVar·SCALAR·FP64·false°matrixVar·MATRIX·FP64°outputVar·MATRIX·FP64";
		BinaryCPInstruction inst = BinaryCPInstruction.parseInstruction(instDefinition);
		testGettingTimeEstimateForCPInst(estimator, inputStats, inst, -1);
	}

	@Test
	public void randCPInstructionExceedMemoryBudgetTest() {
		HashMap<String, VarStats> inputStats = new HashMap<>();
		inputStats.put("matrixVar", generateStats("matrixVar", 1000000, 1000000, -1));
		inputStats.put("outputVar", generateStats("outputVar", 1000000, 1000000, -1));

		String instDefinition = "CP°+°scalarVar·SCALAR·FP64·false°matrixVar·MATRIX·FP64°outputVar·MATRIX·FP64";
		BinaryCPInstruction inst = BinaryCPInstruction.parseInstruction(instDefinition);
		try {
			testGettingTimeEstimateForCPInst(estimator, inputStats, inst, -1);
			Assert.fail("CostEstimationException should have been thrown for the given data size and instruction");
		} catch (CostEstimationException e) {
			// needed catch block to assert that CostEstimationException has been thrown
		}
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Tests for Spark Instructions
	
	@Test
	public void plusBinaryMatrixMatrixSpInstructionTest() throws CostEstimationException {
		HashMap<String, VarStats> inputStats = new HashMap<>();
		inputStats.put("matrixVar", generateStatsWithRdd("matrixVar", 1000000,1000000, 500000000000L));
		inputStats.put("outputVar", generateStats("outputVar", 1000000,1000000, -1));

		String instDefinition = "SPARK°+°scalarVar·SCALAR·FP64·false°matrixVar·MATRIX·FP64°outputVar·MATRIX·FP64";
		BinarySPInstruction inst = BinarySPInstruction.parseInstruction(instDefinition);
		testGettingTimeEstimateForSparkInst(estimator, inputStats, inst, "outputVar", -1);
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Helper methods for testing Instructions

	private VarStats generateStats(String name, long m, long n, long nnz) {
		MatrixCharacteristics mc = new MatrixCharacteristics(m, n, nnz);
		VarStats ret = new VarStats(name, mc);
		long size = OptimizerUtils.estimateSizeExactSparsity(ret.getM(), ret.getN(), ret.getSparsity());
		ret.setAllocatedMemory(size);
		return ret;
	}

	private VarStats generateStatsWithRdd(String name, long m, long n, long nnz) {
		MatrixCharacteristics mc = new MatrixCharacteristics(m, n, nnz);
		VarStats stats = new VarStats(name, mc);
		RDDStats rddStats = new RDDStats(stats);
		stats.setRddStats(rddStats);
		return stats;
	}

	private static void testGettingTimeEstimateForCPInst(
			CostEstimator estimator,
			HashMap<String, VarStats> inputStats,
			CPInstruction targetInstruction,
			double expectedCost
	) throws CostEstimationException {
		if (inputStats != null)
			estimator.putStats(inputStats);
		double actualCost = estimator.getTimeEstimateInst(targetInstruction);

		if (expectedCost < 0) {
			// check error-free cost estimation and meaningful result
			Assert.assertTrue(actualCost > 0);
		} else {
			// check error-free cost estimation and exact result
			Assert.assertEquals(expectedCost, actualCost, 0.0);
		}
	}

	private static void testGettingTimeEstimateForSparkInst(
			CostEstimator estimator,
			HashMap<String, VarStats> inputStats,
			SPInstruction targetInstruction,
			String outputVar,
			double expectedCost
	) throws CostEstimationException {
		if (inputStats != null)
			estimator.putStats(inputStats);
		double actualCost = estimator.getTimeEstimateInst(targetInstruction);
		RDDStats outputRDD = estimator.getStats(outputVar).getRddStats();
		if (outputRDD.isCollected()) {
			// cost directly returned
			if (expectedCost < 0) {
				// check error-free cost estimation and meaningful result
				Assert.assertTrue(actualCost > 0);
			} else {
				// check error-free cost estimation and exact result
				Assert.assertEquals(expectedCost, actualCost, 0.0);
			}
		} else {
			// cost saved in RDD statistics
			double sparkCost = outputRDD.getCost();
			if (expectedCost < 0) {
				// check error-free cost estimation and meaningful result
				Assert.assertTrue(sparkCost > 0);
			} else {
				// check error-free cost estimation and exact result
				Assert.assertEquals(expectedCost, sparkCost, 0.0);
			}
		}
	}
}
