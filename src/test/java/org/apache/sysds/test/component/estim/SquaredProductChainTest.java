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
import org.apache.sysds.hops.estim.EstimatorBasicAvg;
import org.apache.sysds.hops.estim.EstimatorBasicWorst;
import org.apache.sysds.hops.estim.EstimatorBitsetMM;
import org.apache.sysds.hops.estim.EstimatorDensityMap;
import org.apache.sysds.hops.estim.EstimatorLayeredGraph;
import org.apache.sysds.hops.estim.EstimatorMatrixHistogram;
import org.apache.sysds.hops.estim.MMNode;
import org.apache.sysds.hops.estim.SparsityEstimator;
import org.apache.sysds.hops.estim.SparsityEstimator.OpCode;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestUtils;

/**
 * This is a basic sanity check for all estimator, which need
 * to compute a reasonable estimate for uniform data.
 */
public class SquaredProductChainTest extends AutomatedTestBase 
{
	private final static int m = 1000;
	private final static int k = 1000;
	private final static int n = 1000;
	private final static int n2 = 1000;
	private final static double[] case1 = new double[]{0.0001, 0.00007, 0.001};
	private final static double[] case2 = new double[]{0.0006, 0.00007, 0.001};

	private final static double eps1 = 1.0;
	private final static double eps2 = 1e-4;
	private final static double eps3 = 0;
	
	
	@Override
	public void setUp() {
		//do  nothing
	}
	
	@Test
	public void testBasicAvgCase1() {
		runSparsityEstimateTest(new EstimatorBasicAvg(), m, k, n, n2, case1);
	}
	
	@Test
	public void testBasicAvgCase2() {
		runSparsityEstimateTest(new EstimatorBasicAvg(), m, k, n, n2, case2);
	}
	
	@Test
	public void testBasicWorstCase1() {
		runSparsityEstimateTest(new EstimatorBasicWorst(), m, k, n, n2, case1);
	}
	
	@Test
	public void testBasicWorstCase2() {
		runSparsityEstimateTest(new EstimatorBasicWorst(), m, k, n, n2, case2);
	}
	
	@Test
	public void testDensityMapCase1() {
		runSparsityEstimateTest(new EstimatorDensityMap(), m, k, n, n2, case1);
	}
	
	@Test
	public void testDensityMapCase2() {
		runSparsityEstimateTest(new EstimatorDensityMap(), m, k, n, n2, case2);
	}
	
	@Test
	public void testDensityMap7Case1() {
		runSparsityEstimateTest(new EstimatorDensityMap(7), m, k, n, n2, case1);
	}
	
	@Test
	public void testDensityMap7Case2() {
		runSparsityEstimateTest(new EstimatorDensityMap(7), m, k, n, n2, case2);
	}
	
	@Test
	public void testBitsetMatrixCase1() {
		runSparsityEstimateTest(new EstimatorBitsetMM(), m, k, n, n2, case1);
	}
	
	@Test
	public void testBitsetMatrixCase2() {
		runSparsityEstimateTest(new EstimatorBitsetMM(), m, k, n, n2, case2);
	}
	
	@Test
	public void testMatrixHistogramCase1() {
		runSparsityEstimateTest(new EstimatorMatrixHistogram(false), m, k, n, n2, case1);
	}
	
	@Test
	public void testMatrixHistogramCase2() {
		runSparsityEstimateTest(new EstimatorMatrixHistogram(false), m, k, n, n2, case2);
	}
	
	@Test
	public void testMatrixHistogramExceptCase1() {
		runSparsityEstimateTest(new EstimatorMatrixHistogram(true), m, k, n, n2, case1);
	}
	
	@Test
	public void testMatrixHistogramExceptCase2() {
		runSparsityEstimateTest(new EstimatorMatrixHistogram(true), m, k, n, n2, case2);
	}
	
	@Test
	public void testLayeredGraphCase1() {
		runSparsityEstimateTest(new EstimatorLayeredGraph(32), m, k, n, n2, case1);
	}
	
	@Test
	public void testLayeredGraphCase2() {
		runSparsityEstimateTest(new EstimatorLayeredGraph(32), m, k, n, n2, case2);
	}
	
	private static void runSparsityEstimateTest(SparsityEstimator estim, int m, int k, int n, int n2, double[] sp) {
		MatrixBlock m1 = MatrixBlock.randOperations(m, k, sp[0], 1, 1, "uniform", 1);
		MatrixBlock m2 = MatrixBlock.randOperations(k, n, sp[1], 1, 1, "uniform", 2);
		MatrixBlock m3 = MatrixBlock.randOperations(n, n2, sp[2], 1, 1, "uniform", 3);
		MatrixBlock m4 = m1.aggregateBinaryOperations(m1, m2, 
			new MatrixBlock(), InstructionUtils.getMatMultOperator(1));
		MatrixBlock m5 = m4.aggregateBinaryOperations(m4, m3, 
			new MatrixBlock(), InstructionUtils.getMatMultOperator(1));
		
		//compare estimated and real sparsity
		double est = estim.estim(new MMNode(new MMNode(new MMNode(m1), new MMNode(m2),
			OpCode.MM), new MMNode(m3), OpCode.MM)).getSparsity();
		TestUtils.compareScalars(est, m5.getSparsity(),
			(estim instanceof EstimatorBitsetMM) ? eps3 : //exact
			(estim instanceof EstimatorBasicWorst) ? eps1 : eps2);
	}
}
