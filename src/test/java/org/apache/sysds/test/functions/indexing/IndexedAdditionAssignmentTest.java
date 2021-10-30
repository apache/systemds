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

package org.apache.sysds.test.functions.indexing;


import org.junit.Assert;
import org.junit.Test;

import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;

public class IndexedAdditionAssignmentTest extends AutomatedTestBase
{
	private final static String TEST_DIR = "functions/indexing/";
	private final static String TEST_NAME = "IndexedAdditionTest";
	
	private final static String TEST_CLASS_DIR = TEST_DIR + IndexedAdditionAssignmentTest.class.getSimpleName() + "/";
	
	private final static int rows = 1279;
	private final static int cols = 1050;
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"A"}));
	}

	@Test
	public void testIndexedAssignmentAddScalarCP() {
		runIndexedAdditionAssignment(true, ExecType.CP);
	}
	
	@Test
	public void testIndexedAssignmentAddMatrixCP() {
		runIndexedAdditionAssignment(false, ExecType.CP);
	}
	
	@Test
	public void testIndexedAssignmentAddScalarSpark() {
		runIndexedAdditionAssignment(true, ExecType.SPARK);
	}
	
	@Test
	public void testIndexedAssignmentAddMatrixSpark() {
		runIndexedAdditionAssignment(false, ExecType.SPARK);
	}
	
	private void runIndexedAdditionAssignment(boolean scalar, ExecType instType) {
		ExecMode platformOld = setExecMode(instType);
	
		try {
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			loadTestConfiguration(config);
			
			//test is adding or subtracting 7 to area 1x1 or 10x10
			//of an initially constraint (3) matrix and sums it up
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-explain" , "-args",
				Long.toString(rows), Long.toString(cols),
				String.valueOf(scalar).toUpperCase(), output("A")};
			
			runTest(true, false, null, -1);
			
			Double ret = readDMLMatrixFromOutputDir("A").get(new CellIndex(1,1));
			Assert.assertEquals(Double.valueOf(3*rows*cols + 7*(scalar?1:100)),  ret);
		}
		finally {
			resetExecMode(platformOld);
		}
	}
}
