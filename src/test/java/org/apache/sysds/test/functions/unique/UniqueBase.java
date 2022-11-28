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
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

import java.util.HashMap;

public abstract class UniqueBase extends AutomatedTestBase {

	protected abstract String getTestName();

	protected abstract String getTestDir();

	protected abstract String getTestClassDir();

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(getTestName(), new TestConfiguration(getTestClassDir(), getTestName(), new String[] {"A"}));
	}

	protected void uniqueTest(double[][] inputMatrix, double[][] expectedMatrix,
							Types.ExecType instType, double epsilon) {
		Types.ExecMode platformOld = setExecMode(instType);
		try {
			loadTestConfiguration(getTestConfiguration(getTestName()));
			String HOME = SCRIPT_DIR + getTestDir();
			fullDMLScriptName = HOME + getTestName() + ".dml";
			programArgs = new String[]{ "-args", input("I"), output("A")};

			writeInputMatrixWithMTD("I", inputMatrix, true);

			runTest(true, false, null, -1);
			writeExpectedMatrix("A", expectedMatrix);

			compareResultsRowsOutOfOrder(epsilon);
		}
		finally {
			rtplatform = platformOld;
		}
	}
}
