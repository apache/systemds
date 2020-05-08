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

package org.apache.sysds.test.functions.unary.matrix;

import static org.junit.Assert.assertTrue;

import java.util.HashMap;

import org.apache.sysds.common.Types;
import org.apache.sysds.lops.LopProperties;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public class UniqueLength extends AutomatedTestBase {

	private final static String TEST_NAME = "unique_length";
	private final static String TEST_DIR = "functions/unary/matrix/";
	private static final String TEST_CLASS_DIR = TEST_DIR + UniqueLength.class.getSimpleName() + "/";

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"A"}));
	}

	@Test
	public void test1() {
		run_unique_length_test(100, 100, 100, LopProperties.ExecType.CP);
	}

	private void run_unique_length_test(int numberDistinct, int cols, int rows, LopProperties.ExecType instType) {
		Types.ExecMode platformOld = setExecMode(instType);
		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";

			programArgs = new String[] {"-args", String.valueOf(numberDistinct), String.valueOf(cols),
				String.valueOf(rows), output("A")};

			runTest(true, false, null, -1);
			HashMap<CellIndex, Double> frameRead = readDMLMatrixFromHDFS("A");
			for(Double v: frameRead.values()){
				assertTrue("The Unique count does not match: DML:" + v.intValue() + " != actual:" + numberDistinct ,v.intValue() == numberDistinct);
			}
		}
		finally {
			rtplatform = platformOld;
		}
	}
}