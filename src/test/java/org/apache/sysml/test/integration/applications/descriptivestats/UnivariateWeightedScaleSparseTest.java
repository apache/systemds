/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package org.apache.sysml.test.integration.applications.descriptivestats;

import org.junit.Test;

import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;

/**
 * Tests of univariate statistics built-in functions.
 * <p>DOES test optional weight parameters to these functions
 * <p>Tests functions on SPARSE data
 */
public class UnivariateWeightedScaleSparseTest extends UnivariateStatsBase
{
	
	public UnivariateWeightedScaleSparseTest() {
		super();
		TEST_CLASS_DIR = TEST_DIR + UnivariateWeightedScaleSparseTest.class.getSimpleName() + "/";
	}

	// -------------------------------------------------------------------------------------

	@Test
	public void testWeightedScale1() {
		testWeightedScaleWithR(SIZE.DIV4, RANGE.NEG, SPARSITY.SPARSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testWeightedScale2() {
		testWeightedScaleWithR(SIZE.DIV4P1, RANGE.NEG, SPARSITY.SPARSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testWeightedScale3() {
		testWeightedScaleWithR(SIZE.DIV4P3, RANGE.NEG, SPARSITY.SPARSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testWeightedScale4() {
		testWeightedScaleWithR(SIZE.DIV4P3, RANGE.NEG, SPARSITY.SPARSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testWeightedScale5() {
		testWeightedScaleWithR(SIZE.DIV4, RANGE.MIXED, SPARSITY.SPARSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testWeightedScale6() {
		testWeightedScaleWithR(SIZE.DIV4P1, RANGE.MIXED, SPARSITY.SPARSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testWeightedScale7() {
		testWeightedScaleWithR(SIZE.DIV4P3, RANGE.MIXED, SPARSITY.SPARSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testWeightedScale8() {
		testWeightedScaleWithR(SIZE.DIV4P3, RANGE.MIXED, SPARSITY.SPARSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testWeightedScale9() {
		testWeightedScaleWithR(SIZE.DIV4, RANGE.POS, SPARSITY.SPARSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testWeightedScale10() {
		testWeightedScaleWithR(SIZE.DIV4P1, RANGE.POS, SPARSITY.SPARSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testWeightedScale11() {
		testWeightedScaleWithR(SIZE.DIV4P3, RANGE.POS, SPARSITY.SPARSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testWeightedScale12() {
		testWeightedScaleWithR(SIZE.DIV4P3, RANGE.POS, SPARSITY.SPARSE, RUNTIME_PLATFORM.HYBRID);
	}
	
	// -------------------------------------------------------------------------------------
	// Tests 13-24 moved to UnivariateWeightedScaleDenseTest.java
	// -------------------------------------------------------------------------------------
	
	@Test
	public void testWeightedScale25() {
		testWeightedScaleWithR(SIZE.DIV4, RANGE.NEG, SPARSITY.SPARSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testWeightedScale26() {
		testWeightedScaleWithR(SIZE.DIV4P1, RANGE.NEG, SPARSITY.SPARSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testWeightedScale27() {
		testWeightedScaleWithR(SIZE.DIV4P3, RANGE.NEG, SPARSITY.SPARSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testWeightedScale28() {
		testWeightedScaleWithR(SIZE.DIV4P3, RANGE.NEG, SPARSITY.SPARSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testWeightedScale29() {
		testWeightedScaleWithR(SIZE.DIV4, RANGE.MIXED, SPARSITY.SPARSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testWeightedScale30() {
		testWeightedScaleWithR(SIZE.DIV4P1, RANGE.MIXED, SPARSITY.SPARSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testWeightedScale31() {
		testWeightedScaleWithR(SIZE.DIV4P3, RANGE.MIXED, SPARSITY.SPARSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testWeightedScale32() {
		testWeightedScaleWithR(SIZE.DIV4P3, RANGE.MIXED, SPARSITY.SPARSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testWeightedScale33() {
		testWeightedScaleWithR(SIZE.DIV4, RANGE.POS, SPARSITY.SPARSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testWeightedScale34() {
		testWeightedScaleWithR(SIZE.DIV4P1, RANGE.POS, SPARSITY.SPARSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testWeightedScale35() {
		testWeightedScaleWithR(SIZE.DIV4P3, RANGE.POS, SPARSITY.SPARSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testWeightedScale36() {
		testWeightedScaleWithR(SIZE.DIV4P3, RANGE.POS, SPARSITY.SPARSE, RUNTIME_PLATFORM.HADOOP);
	}
	
	// -------------------------------------------------------------------------------------
	// Tests 37-48 moved to UnivariateWeightedScaleDenseTest.java
	// -------------------------------------------------------------------------------------
	
}
