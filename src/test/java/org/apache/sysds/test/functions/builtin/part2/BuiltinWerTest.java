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

import java.util.HashMap;

import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;

public class BuiltinWerTest extends AutomatedTestBase {
	private final static String TEST_NAME = "wer";
	private final static String TEST_DIR = "functions/builtin/";
	private final static String TEST_CLASS_DIR = TEST_DIR + BuiltinWerTest.class.getSimpleName() + "/";

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"W"}));
	}

	@Test
	public void testCP() {
		runWerTest(ExecType.CP);
	}

	@Test
	public void testSpark() {
		runWerTest(ExecType.SPARK);
	}

	private void runWerTest(ExecType instType) {
		ExecMode platformOld = setExecMode(instType);

		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME));

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-explain","-args", output("W")};
			runTest(null);
			
			HashMap<CellIndex, Double> ret = readDMLMatrixFromOutputDir("W");
			Assert.assertEquals(ret.get(new CellIndex(1,1)), 0, 1e-14);
			Assert.assertEquals(ret.get(new CellIndex(2,1)), 1d/4, 1e-14);
			Assert.assertEquals(ret.get(new CellIndex(3,1)), 2d/5, 1e-14);
			Assert.assertEquals(ret.get(new CellIndex(4,1)), 5d/5, 1e-14);
		}
		finally {
			rtplatform = platformOld;
		}
	}
}
