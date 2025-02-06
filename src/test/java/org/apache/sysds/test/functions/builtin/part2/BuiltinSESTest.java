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

package org.apache.sysds.test.functions.builtin.part2;

import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.junit.Assert;
import org.junit.Test;

import java.util.HashMap;

public class BuiltinSESTest extends AutomatedTestBase {
	private final static String TEST_NAME = "ses";
	private final static String TEST_DIR = "functions/builtin/";
	private final static String TEST_CLASS_DIR = TEST_DIR + BuiltinSESTest.class.getSimpleName() + "/";

	private final static int rows = 200;

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[]{"y"}));
	}

	@Test
	public void testSES05() {
		runSESTest(0.5, 199d);
	}
	
	@Test
	public void testSES077() {
		runSESTest(0.77, 199.7013);
	}
	
	@Test
	public void testSES10() {
		runSESTest(1.0, 200d);
	}

	private void runSESTest(double alpha, double expected) {
		loadTestConfiguration(getTestConfiguration(TEST_NAME));
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[] {"-args", 
			String.valueOf(rows), String.valueOf(alpha), output("y")};
		runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);
		HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("y");
		Assert.assertEquals(7, dmlfile.size()); //forecast horizon 7
		Assert.assertEquals(expected, dmlfile.get(new CellIndex(1,1)), 1e-3);
	}
}
