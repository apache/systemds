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

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysml.hops.estim.EstimatorMatrixHistogram.MatrixHistogram;
import org.apache.sysml.hops.estim.SparsityEstimator.OpCode;
import org.apache.sysml.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.test.integration.AutomatedTestBase;

public class MNCReshapeTest extends AutomatedTestBase 
{
	@Override
	public void setUp() {
		//do  nothing
	}
	
	@Test
	public void testMNCReshapeN1() {
		runMNCReshapeTest(1000, 100, 200, 500);
	}
	
	@Test
	public void testMNCReshape1N() {
		runMNCReshapeTest(100, 1000, 500, 200);
	}

	private void runMNCReshapeTest(int m, int n, int m2, int n2) {
		MatrixBlock in = createStructuredInput(m, n, m2, n2);
		MatrixBlock out = LibMatrixReorg.reshape(in, new MatrixBlock(m2, n2, false), m2, n2, true);
		
		MatrixHistogram hIn = new MatrixHistogram(in, false);
		MatrixHistogram hOut = MatrixHistogram.deriveOutputHistogram(
			hIn, null, in.getSparsity(), OpCode.RESHAPE, new long[] {m2,n2});
		
		MatrixHistogram hExpect = new MatrixHistogram(out, false);
		
		//expected exact sparsity, even with sketch propagation
		if( m % m2 == 0 )
			Assert.assertArrayEquals(hExpect.getRowCounts(), hOut.getRowCounts());
		if( n % n2 == 0 )
			Assert.assertArrayEquals(hExpect.getColCounts(), hOut.getColCounts());
	}
	
	private MatrixBlock createStructuredInput(int m, int n, int m2, int n2) {
		if( n % n2 == 0  ) { //1:N
			MatrixBlock tmp = createStructuredInput(n, m, n2, m2);
			return LibMatrixReorg.transpose(tmp, new MatrixBlock(m, n, false));
		}
		else if( m % m2 == 0 ) { //N:1
			MatrixBlock tmp = new MatrixBlock(m, n, false);
			int L = m/m2;
			for(int i=0; i<m; i+=L) {
				for( int k=0; k<L; k++ )
					for(int j=0; j<n/(k+1); j++ ) //j=i/100
						tmp.quickSetValue(i+k, j, 1);
			}
			return tmp;
		}
		else {
			throw new RuntimeException("Unsupported general case.");
		}
	}
}
