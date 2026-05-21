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
import org.apache.sysds.hops.estim.EstimatorMatrixHistogram;
import org.apache.sysds.hops.estim.EstimatorRowWise;
import org.apache.sysds.hops.estim.EstimatorLayeredGraph;
import org.apache.sysds.hops.estim.MMNode;
import org.apache.sysds.hops.estim.SparsityEstimator;
import org.apache.sysds.hops.estim.SparsityEstimator.OpCode;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestUtils;

import java.util.Arrays;
import java.util.Collection;

import org.apache.commons.lang3.NotImplementedException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

/**
 * this is the basic operation check for all estimators with chains of operations including binding operations
 */
@RunWith(value = Parameterized.class)
public class OpBindChainTest extends AutomatedTestBase
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
			{600, 300, 100, new double[]{0.2, 0.4}},
			{600, 200, 300, new double[]{0.1, 0.15}},
		});
	}

	//Average Case
	@Test
	public void testAvgRbind() {
		runSparsityEstimateTest(new EstimatorBasicAvg(), OpCode.RBIND);
	}
	
	@Test
	public void testAvgCbind() {
		runSparsityEstimateTest(new EstimatorBasicAvg(), OpCode.CBIND);
	}
	
	//Worst Case
	@Test
	public void testWorstRbind() {
		runSparsityEstimateTest(new EstimatorBasicWorst(), OpCode.RBIND);
	}
	
	@Test
	public void testWorstCbind() {
		runSparsityEstimateTest(new EstimatorBasicWorst(), OpCode.CBIND);
	}
	
	//DensityMap
	/*@Test
	public void testDMCaserbind() {
		runSparsityEstimateTest(new EstimatorDensityMap(), OpCode.RBIND);
	}
	
	@Test
	public void testDMCasecbind() {
		runSparsityEstimateTest(new EstimatorDensityMap(), OpCode.CBIND);
	}*/
	
	//MNC
	@Test
	public void testMNCRbind() {
		runSparsityEstimateTest(new EstimatorMatrixHistogram(), OpCode.RBIND);
	}
		
	@Test
	public void testMNCCbind() {
		runSparsityEstimateTest(new EstimatorMatrixHistogram(), OpCode.CBIND);
	}

	//Bitset
	@Test
	public void testBitsetCaserbind() {
		runSparsityEstimateTest(new EstimatorBitsetMM(), OpCode.RBIND);
	}
	
	 @Test
	public void testBitsetCasecbind() {
		runSparsityEstimateTest(new EstimatorBitsetMM(), OpCode.CBIND);
	 }
		
	//Layered Graph
	@Test
	public void testLGCaserbind() {
		runSparsityEstimateTest(
			new EstimatorLayeredGraph(EstimatorLayeredGraph.ROUNDS, 7), OpCode.RBIND);
	}
			
	@Test
	public void testLGCasecbind() {
		runSparsityEstimateTest(
			new EstimatorLayeredGraph(EstimatorLayeredGraph.ROUNDS, 3), OpCode.CBIND);
	}

	// Row Wise Sparsity Estimator
	@Test
	public void testRowWiseRbind() {
		runSparsityEstimateTest(new EstimatorRowWise(), OpCode.RBIND);
	}

	@Test
	public void testRowWiseCbind() {
		runSparsityEstimateTest(new EstimatorRowWise(), OpCode.CBIND);
	}


	private void runSparsityEstimateTest(SparsityEstimator estim, OpCode op) {
		MatrixBlock m1 = MatrixBlock.randOperations(m, k, sparsity[0], 1, 1, "uniform", 3);
		MatrixBlock m2;
		MatrixBlock m3 = new MatrixBlock();
		MatrixBlock m4;
		switch(op) {
			case RBIND:
				m2 = MatrixBlock.randOperations(n, k, sparsity[1], 1, 1, "uniform", 7);
				m1.append(m2, m3, false);
				m4 = MatrixBlock.randOperations(k, m, sparsity[1], 1, 1, "uniform", 5);
				break;
			case CBIND:
				m2 = MatrixBlock.randOperations(m, n, sparsity[1], 1, 1, "uniform", 7);
				m1.append(m2, m3, true);
				m4 = MatrixBlock.randOperations(k+n, m, sparsity[1], 1, 1, "uniform", 5);
				break;
			default:
				throw new NotImplementedException();
		}
		MatrixBlock m5 = m3.aggregateBinaryOperations(m3, m4,
				new MatrixBlock(), InstructionUtils.getMatMultOperator(1));
		double est = estim.estim(new MMNode(new MMNode(new MMNode(m1), new MMNode(m2), op), new MMNode(m4), OpCode.MM)).getSparsity();
		//compare estimated and real sparsity
		TestUtils.compareScalars(est, m5.getSparsity(),
			(estim instanceof EstimatorBasicWorst) ? 5e-1 :
			(estim instanceof EstimatorLayeredGraph) ? 5e-2 : 1e-2);
	}
}
