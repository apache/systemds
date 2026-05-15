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

import org.apache.sysds.hops.estim.EstimatorBasicAvg;
import org.apache.sysds.hops.estim.EstimatorBasicWorst;
import org.apache.sysds.hops.estim.EstimatorBitsetMM;
import org.apache.sysds.hops.estim.EstimatorDensityMap;
import org.apache.sysds.hops.estim.EstimatorMatrixHistogram;
import org.apache.sysds.hops.estim.EstimatorRowWise;
import org.apache.sysds.hops.estim.EstimatorLayeredGraph;
import org.apache.sysds.hops.estim.EstimatorSample;
import org.apache.sysds.hops.estim.SparsityEstimator;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestUtils;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Arrays;
import java.util.Collection;

/**
 * This is a basic sanity check for all estimator, which need
 * to compute a reasonable estimate for uniform data.
 */
@RunWith(value = Parameterized.class)
public class SquaredProductTest extends AutomatedTestBase
{
	@Parameterized.Parameter(0)
	public int m;
	@Parameterized.Parameter(1)
	public int k;
	@Parameterized.Parameter(2)
	public int n;
	@Parameterized.Parameter(3)
	public double[] sparsity;
	
	@Override
	public void setUp() {
		//do  nothing
	}

	@Parameterized.Parameters
	public static Collection<Object[]> data() {
		return Arrays.asList(new Object[][] {
			// {m, k, n, sparsity}
			{1000, 1000, 1000, new double[]{0.0001, 0.00007}},
			{1000, 1000, 1000, new double[]{0.0006, 0.00007}},
		});
	}
	
	@Test
	public void testBasicAvg() {
		runSparsityEstimateTest(new EstimatorBasicAvg());
	}
	
	@Test
	public void testBasicWorst() {
		runSparsityEstimateTest(new EstimatorBasicWorst());
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
	public void testMatrixHistogram() {
		runSparsityEstimateTest(new EstimatorMatrixHistogram(false));
	}
	
	@Test
	public void testMatrixHistogramExcept() {
		runSparsityEstimateTest(new EstimatorMatrixHistogram(true));
	}

	@Test
	public void testSampling() {
		runSparsityEstimateTest(new EstimatorSample());
	}
	
	@Test
	public void testSamplingFrac20() {
		runSparsityEstimateTest(new EstimatorSample(0.2));
	}

	@Test
	public void testLayeredGraph() {
		runSparsityEstimateTest(new EstimatorLayeredGraph());
	}

	@Test
	public void testRowWise() {
		runSparsityEstimateTest(new EstimatorRowWise());
	}

	private void runSparsityEstimateTest(SparsityEstimator estim) {
		MatrixBlock m1 = MatrixBlock.randOperations(m, k, sparsity[0], 1, 1, "uniform", 3);
		MatrixBlock m2 = MatrixBlock.randOperations(k, n, sparsity[1], 1, 1, "uniform", 7);
		MatrixBlock m3 = m1.aggregateBinaryOperations(m1, m2, 
			new MatrixBlock(), InstructionUtils.getMatMultOperator(1));
		
		//compare estimated and real sparsity
		double est = estim.estim(m1, m2);
		TestUtils.compareScalars(est, m3.getSparsity(),
			(estim instanceof EstimatorBitsetMM) ? 0 : //exact
			(estim instanceof EstimatorBasicWorst) ? 0.05 : 1e-4);
	}
}
