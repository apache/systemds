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

package org.apache.sysml.test.integration.functions.estim;

import org.junit.Test;
import org.apache.directory.api.util.exception.NotImplementedException;
import org.apache.sysml.hops.estim.EstimatorBasicAvg;
import org.apache.sysml.hops.estim.EstimatorBasicWorst;
import org.apache.sysml.hops.estim.EstimatorBitsetMM;
import org.apache.sysml.hops.estim.EstimatorDensityMap;
import org.apache.sysml.hops.estim.SparsityEstimator;
import org.apache.sysml.hops.estim.SparsityEstimator.OpCode;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.utils.TestUtils;

/**
 * this is the basic operation check for all estimators with single operations
 */
public class OpSingleTest extends AutomatedTestBase 
{
	private final static int m = 600;
	private final static int k = 300;
	private final static double sparsity = 0.2;
//	private final static OpCode eqzero = OpCode.EQZERO;
//	private final static OpCode diag = OpCode.DIAG;
	private final static OpCode neqzero = OpCode.NEQZERO;
	private final static OpCode trans = OpCode.TRANS;
	private final static OpCode reshape = OpCode.RESHAPE;

	@Override
	public void setUp() {
		//do  nothing
	}
	
	@Test
	public void testAvgNeqzero() {
		runSparsityEstimateTest(new EstimatorBasicAvg(), m, k, sparsity, neqzero);
	}
	
	@Test
	public void testAvgTrans() {
		runSparsityEstimateTest(new EstimatorBasicAvg(), m, k, sparsity, trans);
	}
	
	@Test
	public void testAvgReshape() {
		runSparsityEstimateTest(new EstimatorBasicAvg(), m, k, sparsity, reshape);
	}

	@Test
	public void testWCNeqzero() {
		runSparsityEstimateTest(new EstimatorBasicWorst(), m, k, sparsity, neqzero);
	}
	
	@Test
	public void testWCTrans() {
		runSparsityEstimateTest(new EstimatorBasicWorst(), m, k, sparsity, trans);
	}
	
	@Test
	public void testWCReshape() {
		runSparsityEstimateTest(new EstimatorBasicWorst(), m, k, sparsity, reshape);
	} 
	
	@Test
	public void testDMapNeqzero() {
		runSparsityEstimateTest(new EstimatorDensityMap(), m, k, sparsity, neqzero);
	}
	
	@Test
	public void testDMapTrans() {
		runSparsityEstimateTest(new EstimatorDensityMap(), m, k, sparsity, trans);
	}
	
	@Test
	public void testDMapReshape() {
		runSparsityEstimateTest(new EstimatorDensityMap(), m, k, sparsity, reshape);
	}	
	
	@Test
	public void testBitsetNeqzero() {
		runSparsityEstimateTest(new EstimatorBitsetMM(), m, k, sparsity, neqzero);
	}
	
	@Test
	public void testBitsetTrans() {
		runSparsityEstimateTest(new EstimatorBitsetMM(), m, k, sparsity, trans);
	}
	
	@Test
	public void testBitsetReshape() {
		runSparsityEstimateTest(new EstimatorBitsetMM(), m, k, sparsity, reshape);
	}
	
	private void runSparsityEstimateTest(SparsityEstimator estim, int m, int k, double sp, OpCode op) {
		MatrixBlock m1 = MatrixBlock.randOperations(m, k, sp, 1, 1, "uniform", 3);
		MatrixBlock m2 = new MatrixBlock();
		double est = 0;
		switch(op) {
			case EQZERO:
				//TODO find out how to do eqzero
			case DIAG:
			case NEQZERO:
				m2 = m1;
				est = estim.estim(m1, op);
				break;
			case TRANS:
				m2 = m1;
				est = estim.estim(m1, op);
				break;
			case RESHAPE:
				m2 = m1;
				est = estim.estim(m1, op);
				break;
			default:
				throw new NotImplementedException();
		}
		//compare estimated and real sparsity
		TestUtils.compareScalars(est, m2.getSparsity(), 0);
	}
}
