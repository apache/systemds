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

package org.apache.sysds.test.functions.rewrite;

import org.apache.sysds.common.Opcodes;
import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

public class RewriteConstantConjunctionDisjunctionTest extends AutomatedTestBase {

	private static final String TEST_NAME_AND = "RewriteBooleanSimplificationTestAnd";
	private static final String TEST_NAME_OR = "RewriteBooleanSimplificationTestOr";
	private static final String TEST_DIR = "functions/rewrite/";
	private static final String TEST_CLASS_DIR = TEST_DIR + RewriteConstantConjunctionDisjunctionTest.class.getSimpleName() + "/";

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME_AND, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME_AND, new String[] {"R"}));
		addTestConfiguration(TEST_NAME_OR, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME_OR, new String[] {"R"}));
	}

	@Test
	public void testBooleanRewriteAnd() {
		testRewriteBooleanSimplification(TEST_NAME_AND, ExecType.CP, 0.0);
	}

	@Test
	public void testBooleanRewriteOr() {
		testRewriteBooleanSimplification(TEST_NAME_OR, ExecType.CP, 1.0);
	}

	private void testRewriteBooleanSimplification(String testname, ExecType et, double expected) {
		ExecMode platformOld = setExecMode(et);
		
		try {
			TestConfiguration config = getTestConfiguration(testname);
			loadTestConfiguration(config);

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[]{"-stats", "-explain", "-args", output("R")};

			runTest(true, false, null, -1);

			Double ret = readDMLMatrixFromOutputDir("R").get(new CellIndex(1,1));
			if( ret == null )
				ret = 0d;
			Assert.assertEquals(
				"Expected boolean simplification result does not match",
				expected, ret, 0.0001);
			Assert.assertFalse(heavyHittersContainsString(Opcodes.NOT.toString()));
		}
		finally {
			resetExecMode(platformOld);
		}
	}
}
