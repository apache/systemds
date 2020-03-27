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

import org.junit.Test;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.hops.estim.EstimationUtils;
import org.apache.sysds.hops.estim.EstimatorBasicAvg;
import org.apache.sysds.hops.estim.EstimatorBasicWorst;
import org.apache.sysds.hops.estim.EstimatorBitsetMM;
import org.apache.sysds.hops.estim.EstimatorDensityMap;
import org.apache.sysds.hops.estim.EstimatorMatrixHistogram;
import org.apache.sysds.hops.estim.EstimatorSample;
import org.apache.sysds.hops.estim.SparsityEstimator;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestUtils;

public class SelfProductTest extends AutomatedTestBase 
{
	private final static int m = 2500;
	private final static double sparsity1 = 0.0001;
	private final static double sparsity2 = 0.000001;
	private final static double eps1 = 0.05;
	private final static double eps2 = 1e-4;
	private final static double eps3 = 0;
	
	
	@Override
	public void setUp() {
		//do  nothing
	}
	
	@Test
	public void testBasicAvgCase1() {
		runSparsityEstimateTest(new EstimatorBasicAvg(), m, sparsity1);
	}
	
	@Test
	public void testBasicAvgCase2() {
		runSparsityEstimateTest(new EstimatorBasicAvg(), m, sparsity2);
	}
	
	@Test
	public void testDensityMapCase1() {
		runSparsityEstimateTest(new EstimatorDensityMap(), m, sparsity1);
	}
	
	@Test
	public void testDensityMapCase2() {
		runSparsityEstimateTest(new EstimatorDensityMap(), m, sparsity2);
	}
	
	@Test
	public void testDensityMap7Case1() {
		runSparsityEstimateTest(new EstimatorDensityMap(7), m, sparsity1);
	}
	
	@Test
	public void testDensityMap7Case2() {
		runSparsityEstimateTest(new EstimatorDensityMap(7), m, sparsity2);
	}
	
	@Test
	public void testBitsetMatrixCase1() {
		runSparsityEstimateTest(new EstimatorBitsetMM(), m, sparsity1);
	}
	
	@Test
	public void testBitsetMatrixCase2() {
		runSparsityEstimateTest(new EstimatorBitsetMM(), m, sparsity2);
	}
	
	@Test
	public void testMatrixHistogramCase1() {
		runSparsityEstimateTest(new EstimatorMatrixHistogram(false), m, sparsity1);
	}
	
	@Test
	public void testMatrixHistogramCase2() {
		runSparsityEstimateTest(new EstimatorMatrixHistogram(false), m, sparsity2);
	}
	
	@Test
	public void testMatrixHistogramExceptCase1() {
		runSparsityEstimateTest(new EstimatorMatrixHistogram(true), m, sparsity1);
	}
	
	@Test
	public void testMatrixHistogramExceptCase2() {
		runSparsityEstimateTest(new EstimatorMatrixHistogram(true), m, sparsity2);
	}
	
	@Test
	public void testSamplingDefCase1() {
		runSparsityEstimateTest(new EstimatorSample(), m, sparsity1);
	}
	
	@Test
	public void testSamplingDefCase2() {
		runSparsityEstimateTest(new EstimatorSample(), m, sparsity2);
	}
	
	@Test
	public void testSampling20Case1() {
		runSparsityEstimateTest(new EstimatorSample(0.2), m, sparsity1);
	}
	
	@Test
	public void testSampling20Case2() {
		runSparsityEstimateTest(new EstimatorSample(0.2), m, sparsity2);
	}
	
	private static void runSparsityEstimateTest(SparsityEstimator estim, int n, double sp) {
		MatrixBlock m1 = MatrixBlock.randOperations(m, n, sp, 1, 1, "uniform", 3);
		MatrixBlock m3 = m1.aggregateBinaryOperations(m1, m1, 
			new MatrixBlock(), InstructionUtils.getMatMultOperator(1));
		double spExact = OptimizerUtils.getSparsity(m, m,
			EstimationUtils.getSelfProductOutputNnz(m1));
		
		//compare estimated and real sparsity
		double est = estim.estim(m1, m1);
		TestUtils.compareScalars(est, m3.getSparsity(),
			(estim instanceof EstimatorBitsetMM) ? eps3 : //exact
			(estim instanceof EstimatorBasicWorst) ? eps1 : eps2);
		TestUtils.compareScalars(m3.getSparsity(), spExact, eps3);
	}
}
