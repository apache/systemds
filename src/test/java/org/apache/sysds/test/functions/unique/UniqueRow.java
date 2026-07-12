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

package org.apache.sysds.test.functions.unique;

import org.apache.sysds.common.Types;
import org.junit.Test;

public class UniqueRow extends UniqueBase {
	private final static String TEST_NAME = "uniqueRow";
	private final static String TEST_DIR = "functions/unique/";
	private static final String TEST_CLASS_DIR = TEST_DIR + UniqueRow.class.getSimpleName() + "/";

	@Override
	protected String getTestName() {
		return TEST_NAME;
	}

	@Override
	protected String getTestDir() {
		return TEST_DIR;
	}

	@Override
	protected String getTestClassDir() {
		return TEST_CLASS_DIR;
	}

	@Test
	public void testBaseCaseCP() {
		double[][] inputMatrix = {{0}};
		double[][] expectedMatrix = {{0}};
		uniqueTest(inputMatrix, expectedMatrix, Types.ExecType.CP, 0.0);
	}

	@Test
	public void testSkinnyCP() {
		double[][] inputMatrix = {{1,1,6,9,4,2,0,9,0,0,4,4}};
		double[][] expectedMatrix = {{1,6,9,4,2,0}};
		uniqueTest(inputMatrix, expectedMatrix, Types.ExecType.CP, 0.0);
	}

	@Test
	public void testSquareCP() {
		double[][] inputMatrix = {{1, 4, 1}, {2, 5, 2}, {3, 6, 3}};
		double[][] expectedMatrix = {{1, 4},{2, 5},{3, 6}};
		uniqueTest(inputMatrix, expectedMatrix, Types.ExecType.CP, 0.0);
	}

	@Test
	public void testWideCP() {
		double[][] inputMatrix = {{1,7,1},{2,8,2},{3,9,3},{4,10,4},{5,11,5},{6,12,6}};
		double[][] expectedMatrix = {{1,7},{2,8},{3,9},{4,10},{5,11},{6,12}};
		uniqueTest(inputMatrix, expectedMatrix, Types.ExecType.CP, 0.0);
	}

	@Test
	public void testNoDuplicatesCP() {
		double[][] inputMatrix = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
		double[][] expectedMatrix = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
		uniqueTest(inputMatrix, expectedMatrix, Types.ExecType.CP, 0.0);
	}
}
