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
 * <p>Tests functions on DENSE data
 */
public class UnivariateWeightedScaleDenseTest extends UnivariateStatsBase
{
	
	// -------------------------------------------------------------------------------------------------------
	// Tests 1-12 moved to UnivariateWeightedScaleSparseTest.java
	// -------------------------------------------------------------------------------------

	@Test
	public void testWeightedScale13() {
		testWeightedScaleWithR(SIZE.DIV4, RANGE.NEG, SPARSITY.DENSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testWeightedScale14() {
		testWeightedScaleWithR(SIZE.DIV4P1, RANGE.NEG, SPARSITY.DENSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testWeightedScale15() {
		testWeightedScaleWithR(SIZE.DIV4P3, RANGE.NEG, SPARSITY.DENSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testWeightedScale16() {
		testWeightedScaleWithR(SIZE.DIV4P3, RANGE.NEG, SPARSITY.DENSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testWeightedScale17() {
		testWeightedScaleWithR(SIZE.DIV4, RANGE.MIXED, SPARSITY.DENSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testWeightedScale18() {
		testWeightedScaleWithR(SIZE.DIV4P1, RANGE.MIXED, SPARSITY.DENSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testWeightedScale19() {
		testWeightedScaleWithR(SIZE.DIV4P3, RANGE.MIXED, SPARSITY.DENSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testWeightedScale20() {
		testWeightedScaleWithR(SIZE.DIV4P3, RANGE.MIXED, SPARSITY.DENSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testWeightedScale21() {
		testWeightedScaleWithR(SIZE.DIV4, RANGE.POS, SPARSITY.DENSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testWeightedScale22() {
		testWeightedScaleWithR(SIZE.DIV4P1, RANGE.POS, SPARSITY.DENSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testWeightedScale23() {
		testWeightedScaleWithR(SIZE.DIV4P3, RANGE.POS, SPARSITY.DENSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testWeightedScale24() {
		testWeightedScaleWithR(SIZE.DIV4P3, RANGE.POS, SPARSITY.DENSE, RUNTIME_PLATFORM.HYBRID);
	}
	
	// -------------------------------------------------------------------------------------
	// Tests 25-36 moved to UnivariateWeightedScaleSparseTest.java
	// -------------------------------------------------------------------------------------

	@Test
	public void testWeightedScale37() {
		testWeightedScaleWithR(SIZE.DIV4, RANGE.NEG, SPARSITY.DENSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testWeightedScale38() {
		testWeightedScaleWithR(SIZE.DIV4P1, RANGE.NEG, SPARSITY.DENSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testWeightedScale39() {
		testWeightedScaleWithR(SIZE.DIV4P3, RANGE.NEG, SPARSITY.DENSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testWeightedScale40() {
		testWeightedScaleWithR(SIZE.DIV4P3, RANGE.NEG, SPARSITY.DENSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testWeightedScale41() {
		testWeightedScaleWithR(SIZE.DIV4, RANGE.MIXED, SPARSITY.DENSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testWeightedScale42() {
		testWeightedScaleWithR(SIZE.DIV4P1, RANGE.MIXED, SPARSITY.DENSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testWeightedScale43() {
		testWeightedScaleWithR(SIZE.DIV4P3, RANGE.MIXED, SPARSITY.DENSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testWeightedScale44() {
		testWeightedScaleWithR(SIZE.DIV4P3, RANGE.MIXED, SPARSITY.DENSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testWeightedScale45() {
		testWeightedScaleWithR(SIZE.DIV4, RANGE.POS, SPARSITY.DENSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testWeightedScale46() {
		testWeightedScaleWithR(SIZE.DIV4P1, RANGE.POS, SPARSITY.DENSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testWeightedScale47() {
		testWeightedScaleWithR(SIZE.DIV4P3, RANGE.POS, SPARSITY.DENSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testWeightedScale48() {
		testWeightedScaleWithR(SIZE.DIV4P3, RANGE.POS, SPARSITY.DENSE, RUNTIME_PLATFORM.HADOOP);
	}
	// -------------------------------------------------------------------------------------
		
}
