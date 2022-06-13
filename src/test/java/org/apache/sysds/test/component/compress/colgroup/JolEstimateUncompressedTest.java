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

package org.apache.sysds.test.component.compress.colgroup;

import java.util.ArrayList;
import java.util.Collection;

import org.apache.sysds.runtime.compress.colgroup.AColGroup.CompressionType;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.TestUtils;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

/**
 * Test for the Uncompressed Col Group, To verify that we estimate the memory usage to be the worst case possible.
 */
@RunWith(value = Parameterized.class)
public class JolEstimateUncompressedTest extends JolEstimateTest {

	@Parameters
	public static Collection<Object[]> data() {

		ArrayList<Object[]> tests = new ArrayList<>();
		
		// single cell
		// tests.add(new Object[] {DataConverter.convertToMatrixBlock(new double[][] {{0}})});
		// tests.add(new Object[] {DataConverter.convertToMatrixBlock(new double[][] {{1}})});
		// tests.add(new Object[] {DataConverter.convertToMatrixBlock(new double[][] {{42151}})});
		
		// Const
		// tests.add(new Object[] {DataConverter.convertToMatrixBlock(new double[][] {{1,1,1}})});
		
		// Empty
		tests.add(new Object[] {TestUtils.generateTestMatrixBlock(1, 1, 0, 0, 0.0, 7)});
		tests.add(new Object[] {TestUtils.generateTestMatrixBlock(1, 10, 0, 0, 0.0, 7)});
		tests.add(new Object[] {TestUtils.generateTestMatrixBlock(1, 100, 0, 0, 0.0, 7)});
		
		// Small
		tests.add(new Object[] {TestUtils.generateTestMatrixBlock(1, 100, 0, 100, 1.0, 7)});
		tests.add(new Object[] {TestUtils.generateTestMatrixBlock(1, 1000, 0, 100, 0.2, 7)});
		
		// Multi column
		tests.add(new Object[] {TestUtils.generateTestMatrixBlock(2, 10, 0, 100, 1.0, 7)});
		tests.add(new Object[] {TestUtils.generateTestMatrixBlock(13, 100, 0, 100, 1.0, 7)});

		// Const multi column
		// tests.add(new Object[] {DataConverter.convertToMatrixBlock(new double[][] {{1,1,1},{1,1,1},{1,1,1}})});
		tests.add(new Object[] {TestUtils.generateTestMatrixBlock(13, 100, 1, 1, 1.0, 7)});
		tests.add(new Object[] {TestUtils.generateTestMatrixBlock(30, 100, 1, 1, 1.0, 7)});
		
		// empty multi column
		tests.add(new Object[] {TestUtils.generateTestMatrixBlock(10, 100, 0, 0, 0.0, 7)});
		tests.add(new Object[] {TestUtils.generateTestMatrixBlock(100, 100, 0, 0, 0.0, 7)});

		// sparse multi column
		tests.add(new Object[] {TestUtils.generateTestMatrixBlock(13, 100, 0, 100, 0.3, 7)});
		tests.add(new Object[] {TestUtils.generateTestMatrixBlock(100, 100, 0, 100, 0.01, 7)});

		return tests;
	}

	public JolEstimateUncompressedTest(MatrixBlock mb) {
		super(mb);
	}

	@Override
	public CompressionType getCT() {
		return unc;
	}

}