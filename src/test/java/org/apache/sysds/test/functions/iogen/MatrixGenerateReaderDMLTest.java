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

package org.apache.sysds.test.functions.iogen;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types;
import org.apache.sysds.conf.CompilerConfig;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

public class MatrixGenerateReaderDMLTest extends AutomatedTestBase {

	private final static String TEST_NAME = "GenerateReaderTest1";
	private final static String TEST_DIR = "functions/iogen/";
	private final static String TEST_CLASS_DIR = TEST_DIR + MatrixGenerateReaderDMLTest.class.getSimpleName() + "/";

	@Override public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_DIR, TEST_NAME, new String[] {"Y"}));
	}

	@Test public void testGenerateReader1() {

		String data = "1,2,3,4,5\n" + "6,7,8,9,10\n" + "11,12,13,14,15";
		double[][] sampleMatrix = new double[][] {{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}};
		runGenerateReaderTest(Types.ExecMode.SINGLE_NODE, false, data, sampleMatrix);
	}

	// Index start from 0
	@Test public void testGenerateReader2() throws Exception {
		String data = "+1 2:3 4:5 6:7\n" + "-1 8:9 10:11";
		double[][] sampleMatrix = new double[][] {{0, 0, 3, 0, 5, 0, 7, 0, 0, 0, 0, +1},
			{0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 11, -1}};
		runGenerateReaderTest(Types.ExecMode.SINGLE_NODE, false, data, sampleMatrix);
	}

	protected void runGenerateReaderTest(Types.ExecMode platform, boolean parallel, String sampleRaw,
		double[][] sampleMatrix) {

		Types.ExecMode oldPlatform = rtplatform;
		rtplatform = platform;

		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if(rtplatform == Types.ExecMode.SPARK)
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;

		boolean oldpar = CompilerConfig.FLAG_PARREADWRITE_TEXT;

		try {
			CompilerConfig.FLAG_PARREADWRITE_TEXT = parallel;

			TestConfiguration config = getTestConfiguration(TEST_NAME);
			loadTestConfiguration(config);

			String HOME = SCRIPT_DIR + TEST_DIR;
			String dataPath = HOME + "data.raw";
			writeRawString(sampleRaw, dataPath);
			writeInputMatrixWithMTD("sample_matrix", sampleMatrix, false);

			fullDMLScriptName = HOME + "MatrixGenerateReaderTest1.dml";
			programArgs = new String[] {"-nvargs", "data=" + dataPath, "sample_raw=" + sampleRaw,
				"sample_matrix=" + input("sample_matrix"), "ncols=" + sampleMatrix[0].length};

			runTest(true, false, null, -1);
		}
		catch(Exception exception) {
			exception.printStackTrace();
		}
		finally {
			rtplatform = oldPlatform;
			CompilerConfig.FLAG_PARREADWRITE_TEXT = oldpar;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}

	private void writeRawString(String raw, String fileName) throws IOException {
		BufferedWriter writer = new BufferedWriter(new FileWriter(fileName));
		writer.write(raw);
		writer.close();
	}
}
