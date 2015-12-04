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

package org.apache.sysml.test.integration.applications.descriptivestats;

import org.junit.Test;

import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;


/**
 * Tests of univariate statistics built-in functions.
 * <p>Does NOT test optional weight parameters to these functions
 * <p>Tests functions on SPARSE data
 */
public class UnivariateUnweightedScaleSparseTest extends UnivariateStatsBase
{
	
	// -------------------------------------------------------------------------------------
	@Test
	public void testScale1() {
		testScaleWithR(SIZE.DIV4, RANGE.NEG, SPARSITY.SPARSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testScale2() {
		testScaleWithR(SIZE.DIV4P1, RANGE.NEG, SPARSITY.SPARSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testScale3() {
		testScaleWithR(SIZE.DIV4P3, RANGE.NEG, SPARSITY.SPARSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testScale4() {
		testScaleWithR(SIZE.DIV4P3, RANGE.NEG, SPARSITY.SPARSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testScale5() {
		testScaleWithR(SIZE.DIV4, RANGE.MIXED, SPARSITY.SPARSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testScale6() {
		testScaleWithR(SIZE.DIV4P1, RANGE.MIXED, SPARSITY.SPARSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testScale7() {
		testScaleWithR(SIZE.DIV4P3, RANGE.MIXED, SPARSITY.SPARSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testScale8() {
		testScaleWithR(SIZE.DIV4P3, RANGE.MIXED, SPARSITY.SPARSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testScale9() {
		testScaleWithR(SIZE.DIV4, RANGE.POS, SPARSITY.SPARSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testScale10() {
		testScaleWithR(SIZE.DIV4P1, RANGE.POS, SPARSITY.SPARSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testScale11() {
		testScaleWithR(SIZE.DIV4P3, RANGE.POS, SPARSITY.SPARSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testScale12() {
		testScaleWithR(SIZE.DIV4P3, RANGE.POS, SPARSITY.SPARSE, RUNTIME_PLATFORM.HYBRID);
	}
	
	// -------------------------------------------------------------------------------------
	// Tests 13-24 inclusive moved to UnivariateUnweightedScaleDenseTest.java
	// -------------------------------------------------------------------------------------
	
	@Test
	public void testScale25() {
		testScaleWithR(SIZE.DIV4, RANGE.NEG, SPARSITY.SPARSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testScale26() {
		testScaleWithR(SIZE.DIV4P1, RANGE.NEG, SPARSITY.SPARSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testScale27() {
		testScaleWithR(SIZE.DIV4P3, RANGE.NEG, SPARSITY.SPARSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testScale28() {
		testScaleWithR(SIZE.DIV4P3, RANGE.NEG, SPARSITY.SPARSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testScale29() {
		testScaleWithR(SIZE.DIV4, RANGE.MIXED, SPARSITY.SPARSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testScale30() {
		testScaleWithR(SIZE.DIV4P1, RANGE.MIXED, SPARSITY.SPARSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testScale31() {
		testScaleWithR(SIZE.DIV4P3, RANGE.MIXED, SPARSITY.SPARSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testScale32() {
		testScaleWithR(SIZE.DIV4P3, RANGE.MIXED, SPARSITY.SPARSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testScale33() {
		testScaleWithR(SIZE.DIV4, RANGE.POS, SPARSITY.SPARSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testScale34() {
		testScaleWithR(SIZE.DIV4P1, RANGE.POS, SPARSITY.SPARSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testScale35() {
		testScaleWithR(SIZE.DIV4P3, RANGE.POS, SPARSITY.SPARSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testScale36() {
		testScaleWithR(SIZE.DIV4P3, RANGE.POS, SPARSITY.SPARSE, RUNTIME_PLATFORM.HADOOP);
	}
	
	
	
	// -------------------------------------------------------------------------------------------------------
	
	
	
}
