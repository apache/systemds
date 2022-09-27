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

package org.apache.sysds.test.functions.io.compressed;

import static org.junit.Assert.fail;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.conf.CompilerConfig;
import org.apache.sysds.runtime.compress.io.WriterCompressed;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public class ReadCompressedTest extends CompressedTestBase {

	private final static String TEST_NAME = "ReadCompressedTest";
	private final static String TEST_CLASS_DIR = TEST_DIR + ReadCompressedTest.class.getSimpleName() + "/";

	private final double expected;

	public ReadCompressedTest() {
		try {
			String HOME = SCRIPT_DIR + TEST_DIR;
			String n = HOME + INPUT_DIR + getInputFileName() + ".cla";
			MatrixBlock mb = TestUtils.ceil(TestUtils.generateTestMatrixBlock(130, 62, 100, 102, 1.0, 215));
			WriterCompressed.writeCompressedMatrixToHDFS(mb, n, 50);
			MatrixCharacteristics mc = new MatrixCharacteristics(130, 62, 50);
			HDFSTool.writeMetaDataFile(n + ".mtd", ValueType.FP64, mc, FileFormat.COMPRESSED);
			expected = mb.sum();
		}
		catch(Exception e) {
			throw new RuntimeException(e);
		}
	}

	protected String getTestName() {
		return TEST_NAME;
	}

	protected String getTestClassDir() {
		return TEST_CLASS_DIR;
	}

	protected String getInputFileName() {
		return "comp";
	}

	@Test
	public void testCSV_Sequential_CP1() {
		runTest(ExecMode.SINGLE_NODE, false);
	}

	@Test
	public void testCSV_Parallel_CP1() {
		runTest(ExecMode.SINGLE_NODE, true);
	}

	@Test
	public void testCSV_Sequential_CP() {
		runTest(ExecMode.HYBRID, false);
	}

	@Test
	public void testCSV_Parallel_CP() {
		runTest(ExecMode.HYBRID, true);
	}

	@Test
	public void testCSV_SP() {
		runTest(ExecMode.SPARK, false);
	}

	protected String runTest(ExecMode platform, boolean parallel) {
		ExecMode oldPlatform = rtplatform;
		rtplatform = platform;

		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if(rtplatform == ExecMode.SPARK)
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;

		boolean oldpar = CompilerConfig.FLAG_PARREADWRITE_TEXT;

		try {
			CompilerConfig.FLAG_PARREADWRITE_TEXT = parallel;

			TestConfiguration config = getTestConfiguration(getTestName());
			loadTestConfiguration(config);
			setOutputBuffering(true); // otherwise NPEs

			String HOME = SCRIPT_DIR + TEST_DIR;
			String n = HOME + INPUT_DIR + getInputFileName() + ".cla";
			String dmlOutput = output("dml.scalar");

			fullDMLScriptName = HOME + getTestName() + ".dml";
			programArgs = new String[] {"-args", n, dmlOutput};

			String out = runTest(true, false, null, -1).toString();
			double dmlScalar = TestUtils.readDMLScalar(dmlOutput);
			TestUtils.compareScalars(dmlScalar, expected, eps);
			return out;
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
		finally {
			rtplatform = oldPlatform;
			CompilerConfig.FLAG_PARREADWRITE_TEXT = oldpar;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}

		return null;
	}

}
