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

package org.apache.sysds.test.component.estim;

import org.junit.Assume;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Arrays;
import java.util.Collection;

import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.hops.estim.EstimationUtils;
import org.apache.sysds.hops.estim.EstimatorBasicAvg;
import org.apache.sysds.hops.estim.EstimatorBasicWorst;
import org.apache.sysds.hops.estim.EstimatorBitsetMM;
import org.apache.sysds.hops.estim.EstimatorDensityMap;
import org.apache.sysds.hops.estim.EstimatorLayeredGraph;
import org.apache.sysds.hops.estim.EstimatorMatrixHistogram;
import org.apache.sysds.hops.estim.EstimatorRowWise;
import org.apache.sysds.hops.estim.EstimatorSample;
import org.apache.sysds.hops.estim.EstimatorSampleRa;
import org.apache.sysds.hops.estim.SparsityEstimator;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestUtils;

@RunWith(value = Parameterized.class)
public class SelfProductTest extends AutomatedTestBase
{
	@Parameterized.Parameter(0)
	public int m;
	@Parameterized.Parameter(1)
	public double sparsity;
	
	@Override
	public void setUp() {
		//do  nothing
	}

	@Parameterized.Parameters
	public static Collection<Object[]> data() {
		return Arrays.asList(new Object[][] {
			// {m, sparsity}
			{625, 0.5},
			{1250, 0.1},
			{2500, 0.0001},
			{2500, 0.000001},
		});
	}

	@Test
	public void testBasicAvg() {
		runSparsityEstimateTest(new EstimatorBasicAvg());
	}
	
	@Test
	public void testDensityMap() {
		runSparsityEstimateTest(new EstimatorDensityMap());
	}
	
	@Test
	public void testDensityMapBlocksize7() {
		runSparsityEstimateTest(new EstimatorDensityMap(7));
	}
	
	@Test
	public void testBitsetMatrix() {
		runSparsityEstimateTest(new EstimatorBitsetMM());
	}
	
	@Test
	public void testBitsetMatrixType2() {
		runSparsityEstimateTest(new EstimatorBitsetMM(2));
	}
	
	@Test
	public void testMatrixHistogram() {
		runSparsityEstimateTest(new EstimatorMatrixHistogram(false));
	}
	
	@Test
	public void testMatrixHistogramExtended() {
		runSparsityEstimateTest(new EstimatorMatrixHistogram(true));
	}
	
	@Test
	public void testSampling() {
		Assume.assumeTrue(sparsity < 0.1);
		runSparsityEstimateTest(new EstimatorSample());
	}
	
	@Test
	public void testSamplingFrac20() {
		Assume.assumeTrue(sparsity < 0.1);
		runSparsityEstimateTest(new EstimatorSample(0.2));
	}
	
	@Test
	public void testSamplingRa() {
		runSparsityEstimateTest(new EstimatorSampleRa());
	}
	
	@Test
	public void testSamplingRaFrac20() {
		runSparsityEstimateTest(new EstimatorSampleRa(0.2));
	}
	
	@Test
	public void testLayeredGraph() {
		runSparsityEstimateTest(new EstimatorLayeredGraph(EstimatorLayeredGraph.ROUNDS, 13));
	}

	@Test
	public void testLayeredGraph64Rounds() {
		runSparsityEstimateTest(new EstimatorLayeredGraph(64, 13));
	}

	@Test
	public void testRowWise() {
		runSparsityEstimateTest(new EstimatorRowWise());
	}

	private void runSparsityEstimateTest(SparsityEstimator estim) {
		MatrixBlock m1 = MatrixBlock.randOperations(m, m, sparsity, 1, 1, "uniform", 3);
		MatrixBlock m3 = m1.aggregateBinaryOperations(m1, m1, 
			new MatrixBlock(), InstructionUtils.getMatMultOperator(1));
		double spExact1 = OptimizerUtils.getSparsity(m, m,
			EstimationUtils.getSelfProductOutputNnz(m1));
		double spExact2 = sparsity<0.4 ? OptimizerUtils.getSparsity(m, m,
			EstimationUtils.getSparseProductOutputNnz(m1, m1)) : spExact1;
		
		//compare estimated and real sparsity
		double est = estim.estim(m1, m1);
		TestUtils.compareScalars(est, m3.getSparsity(),
			(estim instanceof EstimatorBitsetMM) ? 0 : //exact
			(estim instanceof EstimatorBasicWorst || estim instanceof EstimatorLayeredGraph) ? 0.05 :
			(sparsity == 0.1 && estim instanceof EstimatorSampleRa) ? 0.12 : 1e-4);
		TestUtils.compareScalars(m3.getSparsity(), spExact1, 0);
		TestUtils.compareScalars(m3.getSparsity(), spExact2, 0);
	}
}
