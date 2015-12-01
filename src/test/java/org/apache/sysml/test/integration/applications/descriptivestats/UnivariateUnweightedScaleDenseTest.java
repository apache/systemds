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

package com.ibm.bi.dml.test.integration.applications.descriptivestats;

import org.junit.Test;

import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;

/**
 * Tests of univariate statistics built-in functions.
 * <p>Does NOT test optional weight parameters to these functions
 * <p>Tests functions on DENSE data
 */
public class UnivariateUnweightedScaleDenseTest extends UnivariateStatsBase
{
	
	

	// -------------------------------------------------------------------------------------
	// Tests 1-12 moved to UnivariateUnweightedScaleSparseTest.java
	// -------------------------------------------------------------------------------------

	@Test
	public void testScale13() {
		testScaleWithR(SIZE.DIV4, RANGE.NEG, SPARSITY.DENSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testScale14() {
		testScaleWithR(SIZE.DIV4P1, RANGE.NEG, SPARSITY.DENSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testScale15() {
		testScaleWithR(SIZE.DIV4P3, RANGE.NEG, SPARSITY.DENSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testScale16() {
		testScaleWithR(SIZE.DIV4P3, RANGE.NEG, SPARSITY.DENSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testScale17() {
		testScaleWithR(SIZE.DIV4, RANGE.MIXED, SPARSITY.DENSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testScale18() {
		testScaleWithR(SIZE.DIV4P1, RANGE.MIXED, SPARSITY.DENSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testScale19() {
		testScaleWithR(SIZE.DIV4P3, RANGE.MIXED, SPARSITY.DENSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testScale20() {
		testScaleWithR(SIZE.DIV4P3, RANGE.MIXED, SPARSITY.DENSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testScale21() {
		testScaleWithR(SIZE.DIV4, RANGE.POS, SPARSITY.DENSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testScale22() {
		testScaleWithR(SIZE.DIV4P1, RANGE.POS, SPARSITY.DENSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testScale23() {
		testScaleWithR(SIZE.DIV4P3, RANGE.POS, SPARSITY.DENSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testScale24() {
		testScaleWithR(SIZE.DIV4P3, RANGE.POS, SPARSITY.DENSE, RUNTIME_PLATFORM.HYBRID);
	}
	
	// -------------------------------------------------------------------------------------
	// Tests 25-36 moved to UnivariateUnweightedScaleSparseTest.java
	// -------------------------------------------------------------------------------------

	@Test
	public void testScale37() {
		testScaleWithR(SIZE.DIV4, RANGE.NEG, SPARSITY.DENSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testScale38() {
		testScaleWithR(SIZE.DIV4P1, RANGE.NEG, SPARSITY.DENSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testScale39() {
		testScaleWithR(SIZE.DIV4P3, RANGE.NEG, SPARSITY.DENSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testScale40() {
		testScaleWithR(SIZE.DIV4P3, RANGE.NEG, SPARSITY.DENSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testScale41() {
		testScaleWithR(SIZE.DIV4, RANGE.MIXED, SPARSITY.DENSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testScale42() {
		testScaleWithR(SIZE.DIV4P1, RANGE.MIXED, SPARSITY.DENSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testScale43() {
		testScaleWithR(SIZE.DIV4P3, RANGE.MIXED, SPARSITY.DENSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testScale44() {
		testScaleWithR(SIZE.DIV4P3, RANGE.MIXED, SPARSITY.DENSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testScale45() {
		testScaleWithR(SIZE.DIV4, RANGE.POS, SPARSITY.DENSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testScale46() {
		testScaleWithR(SIZE.DIV4P1, RANGE.POS, SPARSITY.DENSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testScale47() {
		testScaleWithR(SIZE.DIV4P3, RANGE.POS, SPARSITY.DENSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testScale48() {
		testScaleWithR(SIZE.DIV4P3, RANGE.POS, SPARSITY.DENSE, RUNTIME_PLATFORM.HADOOP);
	}
	// -------------------------------------------------------------------------------------
		
	
}
