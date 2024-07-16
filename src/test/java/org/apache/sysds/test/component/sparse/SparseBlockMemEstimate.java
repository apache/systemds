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

package org.apache.sysds.test.component.sparse;

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseBlockFactory;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestUtils;

/**
 * This is a sparse matrix block component test for sparse block memory
 * estimation functionality.
 * 
 */
public class SparseBlockMemEstimate extends AutomatedTestBase 
{
	private final static int rows = 320;
	private final static int cols = 130;
	private final static double sparsity1 = 0.39;
	private final static double sparsity2 = 0.0001;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
	}

	@Test
	public void testSparseBlockSparse()  {
		runSparseBlockMemoryTest(sparsity1);
	}
	
	@Test
	public void testSparseBlockUltraSparse()  {
		runSparseBlockMemoryTest(sparsity2);
	}
	
	private static void runSparseBlockMemoryTest( double sparsity)
	{
		double memMCSC = SparseBlockFactory.estimateSizeSparseInMemory(SparseBlock.Type.MCSC, rows, cols, sparsity);
		double memMCSR = SparseBlockFactory.estimateSizeSparseInMemory(SparseBlock.Type.MCSR, rows, cols, sparsity);
		double memCSR = SparseBlockFactory.estimateSizeSparseInMemory(SparseBlock.Type.CSR, rows, cols, sparsity);
		double memCOO = SparseBlockFactory.estimateSizeSparseInMemory(SparseBlock.Type.COO, rows, cols, sparsity);
		double memDCSR = SparseBlockFactory.estimateSizeSparseInMemory(SparseBlock.Type.DCSR, rows, cols, sparsity);
		double memDense = MatrixBlock.estimateSizeDenseInMemory(rows, cols);
		
		//check negative estimate
		if( memMCSC <= 0 )
			Assert.fail("SparseBlockMCSC memory estimate <= 0.");
		if( memMCSR <= 0 )
			Assert.fail("SparseBlockMCSR memory estimate <= 0.");
		if( memCSR  <= 0 )
			Assert.fail("SparseBlockCSR memory estimate <= 0.");
		if( memCOO  <= 0 )
			Assert.fail("SparseBlockCOO memory estimate <= 0.");
		if( memDCSR  <= 0 )
			Assert.fail("SparseBlockDCSR memory estimate <= 0.");
		
		//check dense estimate
		if( memMCSC > memDense )
			Assert.fail("SparseBlockMCSC memory estimate larger than dense estimate.");
		if( memMCSR > memDense )
			Assert.fail("SparseBlockMCSR memory estimate larger than dense estimate.");
		if( memCSR > memDense )
			Assert.fail("SparseBlockCSR memory estimate larger than dense estimate.");
		if( memCOO > memDense )
			Assert.fail("SparseBlockCOO memory estimate larger than dense estimate.");
		if( memDCSR > memDense )
			Assert.fail("SparseBlockDCSR memory estimate larger than dense estimate.");
		
		//check sparse estimates relations
		if( sparsity == sparsity1 ) { //sparse (pref CSR)
			if( memMCSC < memCSR )
				Assert.fail("SparseBlockMCSC memory estimate smaller than SparseBlockCSR estimate.");
			if( memMCSR < memCSR )
				Assert.fail("SparseBlockMCSR memory estimate smaller than SparseBlockCSR estimate.");
			if( memCOO < memCSR )
				Assert.fail("SparseBlockCOO memory estimate smaller than SparseBlockCSR estimate.");
		}
		else { //ultra-sparse (pref COO)
			if( memMCSC < memCOO )
				Assert.fail("SparseBlockMCSC memory estimate smaller than SparseBlockCOO estimate.");
			if( memMCSR < memCOO )
				Assert.fail("SparseBlockMCSR memory estimate smaller than SparseBlockCOO estimate.");
			if( memCSR < memCOO )
				Assert.fail("SparseBlockCSR memory estimate smaller than SparseBlockCOO estimate.");
		}
	}
}
