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

package org.apache.sysds.test.functions.misc;

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

public class ListAppendSizeTest extends AutomatedTestBase
{
	private static final String TEST_NAME1 = "ListAppendSize1";
	private static final String TEST_NAME2 = "ListAppendSize2";
	private static final String TEST_NAME3 = "ListAppendSize3";
	private static final String TEST_NAME4 = "ListAppendSize4";
	
	private static final String TEST_DIR = "functions/misc/";
	private static final String TEST_CLASS_DIR = TEST_DIR + ListAppendSizeTest.class.getSimpleName() + "/";
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration( TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "R" }) );
		addTestConfiguration( TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] { "R" }) );
		addTestConfiguration( TEST_NAME3, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3, new String[] { "R" }) );
		addTestConfiguration( TEST_NAME4, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME4, new String[] { "R" }) );
	}
	
	@Test
	public void testListAppendSize1CP() {
		runListAppendSize(TEST_NAME1, ExecType.CP, 4);
	}
	
	@Test
	public void testListAppendSize2CP() {
		runListAppendSize(TEST_NAME2, ExecType.CP, 3);
	}
	
	@Test
	public void testListAppendSize3CP() {
		runListAppendSize(TEST_NAME3, ExecType.CP, 2);
	}
	
	@Test
	public void testListAppendSize4CP() {
		runListAppendSize(TEST_NAME4, ExecType.CP, 4);
	}
	
	private void runListAppendSize(String testname, ExecType type, int expected) {
		ExecMode platformOld = setExecMode(type);
		
		try {
			getAndLoadTestConfiguration(testname);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[]{ "-stats","-explain","-args", output("R") };
			
			//run test
			runTest(true, false, null, -1);
			double ret = readDMLMatrixFromOutputDir("R").get(new CellIndex(1,1));
			Assert.assertEquals(Integer.valueOf(expected), Integer.valueOf((int)ret));
		}
		finally {
			resetExecMode(platformOld);
		}
	}
}
