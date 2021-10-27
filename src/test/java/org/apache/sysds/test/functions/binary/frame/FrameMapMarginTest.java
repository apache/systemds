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

package org.apache.sysds.test.functions.binary.frame;

import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

public class FrameMapMarginTest extends AutomatedTestBase {
	private final static String TEST_NAME = "mapMargin";
	private final static String TEST_DIR = "functions/binary/frame/";
	private static final String TEST_CLASS_DIR = TEST_DIR + FrameMapMarginTest.class.getSimpleName() + "/";

	private final static int rows = 100;
	private final static Types.ValueType[] schemaStrings1 = {Types.ValueType.STRING, Types.ValueType.STRING};

	@BeforeClass
	public static void init() {
		TestUtils.clearDirectory(TEST_DATA_DIR + TEST_CLASS_DIR);
	}

	@AfterClass
	public static void cleanUp() {
		if (TEST_CACHE_ENABLED) {
			TestUtils.clearDirectory(TEST_DATA_DIR + TEST_CLASS_DIR);
		}
	}

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"D"}));
		if (TEST_CACHE_ENABLED) {
			setOutAndExpectedDeletionDisabled(true);
		}
	}

	@Test
	public void testMarginColCP() {
		runDmlMapTest("x -> UtilFunctions.concat(x)", 2, ExecType.CP);
	}

	@Test
	public void testMarginColSP() {
		runDmlMapTest("x -> UtilFunctions.concat(x)", 2, ExecType.CP);
	}

	@Test
	public void testMarginRowCP() {
		runDmlMapTest("x -> UtilFunctions.concat(x)", 1, ExecType.SPARK);
	}

	@Test
	public void testMarginRowSP() {
		runDmlMapTest("x -> UtilFunctions.concat(x)", 1, ExecType.SPARK);
	}

	private void runDmlMapTest( String expression, int margin, ExecType et)
	{
		Types.ExecMode platformOld = setExecMode(et);

		try {
			getAndLoadTestConfiguration(TEST_NAME);

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] { "-stats","-args", input("A"), expression, String.valueOf(margin), output("O"), output("I")};

			double[][] A = getRandomMatrix(rows, 2, 0, 1, 1, 2);
			writeInputFrameWithMTD("A", A, true, schemaStrings1, FileFormat.CSV);
			
			runTest(true, false, null, -1);

			FrameBlock outputFrame = readDMLFrameFromHDFS("O", FileFormat.CSV);
			FrameBlock inputFrame = readDMLFrameFromHDFS("I", FileFormat.CSV);

			String[][] inputDerived, outputDerived;
			if (margin == 1) {
				inputDerived = new String[rows][1];
				outputDerived = new String[rows][1];
				for(int i = 0; i < rows; i++) {
					inputDerived[i][0] = ((String[])inputFrame.getColumnData(0))[i] + ((String[])inputFrame.getColumnData(1))[i];
					outputDerived[i][0] = ((String[])outputFrame.getColumnData(0))[i];
				}
				TestUtils.compareFrames(inputDerived, outputDerived, rows, 1);
			} else {
				inputDerived = new String[1][2];
				outputDerived = new String[1][2];
				inputDerived[0][0] = UtilFunctions.concat((String[])inputFrame.getColumnData(0));
				inputDerived[0][1] = UtilFunctions.concat((String[])inputFrame.getColumnData(1));
				outputDerived[0][0] = ((String[])outputFrame.getColumnData(0))[0];
				outputDerived[0][1] = ((String[])outputFrame.getColumnData(1))[0];
				TestUtils.compareFrames(inputDerived, outputDerived, 1, 2);
			}
		}
		catch (Exception ex) {
			throw new RuntimeException(ex);
		}
		finally {
			resetExecMode(platformOld);
		}
	}
}
