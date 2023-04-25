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

package org.apache.sysds.test.component.compress.plan;

import static org.junit.Assert.fail;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.colgroup.AColGroup.CompressionType;
import org.apache.sysds.runtime.compress.plan.CompressionPlanFactory;
import org.apache.sysds.runtime.compress.plan.IPlanEncode;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public class CustomEncodePlanTest {
	protected static final Log LOG = LogFactory.getLog(CustomEncodePlanTest.class.getName());

	@Test
	public void testEncodeDDCConst() {
		testDDCSingle(new MatrixBlock(10, 2, 3.0));
	}

	@Test
	public void testEncodeDDCNonZero1() {
		testDDCSingle(TestUtils.generateTestMatrixBlock(10, 3, 1, 1, 0.5, 235));
	}

	@Test
	public void testEncodeDDCNonZero2() {
		testDDCSingle(TestUtils.generateTestMatrixBlock(10, 3, 1, 1, 0.9, 235));
	}

	@Test
	public void testEncodeDDCConstTwo() {
		testDDCTwo(new MatrixBlock(10, 2, 3.0));
	}

	@Test
	public void testEncodeDDCNonZeroTwo1() {
		testDDCTwo(TestUtils.generateTestMatrixBlock(10, 3, 1, 1, 0.5, 235));
	}

	@Test
	public void testEncodeDDCNonZeroTwo2() {
		testDDCTwo(TestUtils.generateTestMatrixBlock(10, 3, 1, 1, 0.9, 235));
	}

	@Test
	public void testEncodeDDCNonZeroTwo3() {
		testDDCTwo(TestUtils.generateTestMatrixBlock(10, 3, 1, 1, 0.1, 235));
	}

	@Test
	public void testEncodeDDCNonZeroTwo4() {
		testDDCTwo(TestUtils.generateTestMatrixBlock(10, 3, 1, 1, 1.0, 235));
	}

	@Test
	public void testEncodeDDCNonZeroTwo5() {
		testDDCTwo(TestUtils.generateTestMatrixBlock(10, 3, 1, 1, 0.0, 235));
	}

	@Test
	public void testEncodeDDCConstN() {
		testDDCn(new MatrixBlock(10, 2, 3.0), 3);
	}

	@Test
	public void testEncodeDDCNonZeroN1() {
		testDDCn(TestUtils.generateTestMatrixBlock(10, 5, 1, 1, 0.5, 235), 3);
	}

	@Test
	public void testEncodeDDCNonZeroN2() {
		testDDCn(TestUtils.generateTestMatrixBlock(10, 4, 1, 1, 0.9, 235), 3);
	}

	private void testDDCSingle(MatrixBlock mb) {
		try {

			IPlanEncode plan = CompressionPlanFactory.singleCols(mb.getNumColumns(), CompressionType.DDC, 1);
			testPlan(mb, plan);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	protected void testDDCTwo(MatrixBlock mb) {
		try {

			IPlanEncode plan = CompressionPlanFactory.twoCols(mb.getNumColumns(), CompressionType.DDC, 1);
			testPlan(mb, plan);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	protected void testDDCn(MatrixBlock mb, int n) {
		try {
			IPlanEncode plan = CompressionPlanFactory.nCols(mb.getNumColumns(), n, CompressionType.DDC, 1);
			testPlan(mb, plan);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	protected void testPlan(MatrixBlock mb, IPlanEncode plan) {
		try {

			plan.expandPlan(mb);
			MatrixBlock cmb = plan.encode(mb);
			TestUtils.compareMatrices(mb, cmb, 0);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}
}
