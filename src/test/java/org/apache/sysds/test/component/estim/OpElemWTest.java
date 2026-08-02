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
import org.apache.sysds.hops.estim.SparsityEstimator.OpCode;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestUtils;

import java.util.Arrays;
import java.util.Collection;

import org.apache.commons.lang3.NotImplementedException;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

/**
 * this is the basic operation check for all estimators with element-wise operations
 */
@RunWith(value = Parameterized.class)
public class OpElemWTest extends AutomatedTestBase 
{
	@Parameterized.Parameter(0)
	public int m;
	@Parameterized.Parameter(1)
	public int n;
	@Parameterized.Parameter(2)
	public double[] sparsity;

	@Override
	public void setUp() {
		//do  nothing
	}

	@Parameterized.Parameters
	public static Collection<Object[]> data() {
		return Arrays.asList(new Object[][] {
			// {m, n, sparsity}
			{1600, 700, new double[]{0.2, 0.4}},
			{900, 1200, new double[]{0.01, 0.125}},
		});
	}

	//Average Case
	@Test
	public void testAvgMult() {
		runSparsityEstimateTest(new EstimatorBasicAvg(), OpCode.MULT);
	}
	
	@Test
	public void testAvgPlus() {
		runSparsityEstimateTest(new EstimatorBasicAvg(), OpCode.PLUS);
	}
	
	//Worst Case
	@Test
	public void testWorstMult() {
		runSparsityEstimateTest(new EstimatorBasicWorst(), OpCode.MULT);
	}
	
	@Test
	public void testWorstPlus() {
		runSparsityEstimateTest(new EstimatorBasicWorst(), OpCode.PLUS);
	}
	
	//DensityMap
	@Test
	public void testDMMult() {
		runSparsityEstimateTest(new EstimatorDensityMap(), OpCode.MULT);
	}
	
	@Test
	public void testDMPlus() {
		runSparsityEstimateTest(new EstimatorDensityMap(), OpCode.PLUS);
	}
	
	//MNC
	@Test
	public void testMNCMult() {
		runSparsityEstimateTest(new EstimatorMatrixHistogram(), OpCode.MULT);
	}
	
	@Test
	public void testMNCPlus() {
		runSparsityEstimateTest(new EstimatorMatrixHistogram(), OpCode.PLUS);
	}
	
	//Bitset
	@Test
	public void testBitsetMult() {
		runSparsityEstimateTest(new EstimatorBitsetMM(), OpCode.MULT);
	}
	
	@Test
	public void testBitsetPlus() {
		runSparsityEstimateTest(new EstimatorBitsetMM(), OpCode.PLUS);
	}

	//Layered Graph
	@Test
	public void testLGCasemult() {
		runSparsityEstimateTest(new EstimatorLayeredGraph(), OpCode.MULT);
	}
	
	@Test
	public void testLGCaseplus() {
		runSparsityEstimateTest(new EstimatorLayeredGraph(), OpCode.PLUS);
	}
	
	//Sample
	@Test
	public void testSampleMult() {
		runSparsityEstimateTest(new EstimatorSample(), OpCode.MULT);
	}
	
	@Test
	public void testSamplePlus() {
		runSparsityEstimateTest(new EstimatorSample(), OpCode.PLUS);
	}

	// Row Wise Sparsity Estimator
	@Test
	public void testRowWiseMult() {
		runSparsityEstimateTest(new EstimatorRowWise(), OpCode.MULT);
	}

	@Test
	public void testRowWisePlus() {
		runSparsityEstimateTest(new EstimatorRowWise(), OpCode.PLUS);
	}

	private void runSparsityEstimateTest(SparsityEstimator estim, OpCode op) {
		MatrixBlock m1 = MatrixBlock.randOperations(m, n, sparsity[0], 1, 1, "uniform", 3);
		MatrixBlock m2 = MatrixBlock.randOperations(m, n, sparsity[1], 1, 1, "uniform", 7);
		MatrixBlock m3 = new MatrixBlock();
		BinaryOperator bOp;
		switch(op) {
			case MULT:
				bOp = new BinaryOperator(Multiply.getMultiplyFnObject());
				break;
			case PLUS:
				bOp = new BinaryOperator(Plus.getPlusFnObject());
				break;
				default:
					throw new NotImplementedException();
		}
		m1.binaryOperations(bOp, m2, m3);
		double est = estim.estim(m1, m2, op);
		//compare estimated and real sparsity
		TestUtils.compareScalars(est, m3.getSparsity(), (estim instanceof EstimatorBasicWorst) ? 5e-1 :
			(estim instanceof EstimatorLayeredGraph) ? 3e-2 : 5e-3);
	}
}
