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

import java.io.IOException;

import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.test.component.compress.io.IOCompressionTestUtils;
import org.junit.Test;

/**
 * JUnit Test cases to evaluate the functionality of reading CSV files.
 * 
 * Test 1: write() w/ all properties. Test 2: read(format="csv") w/o mtd file. Test 3: read() w/ complete mtd file.
 *
 */

public class WriteCompressedTest extends AutomatedTestBase {

	private final static String TEST_NAME = "WriteCompressedTest";
	private final static String TEST_DIR = "functions/io/compressed/";
	private final static String TEST_CLASS_DIR = TEST_DIR + WriteCompressedTest.class.getSimpleName() + "/";

	private final static double eps = 1e-9;

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"Rout"}));
	}

	@Test
	public void testCP() throws IOException {
		runWriteTest(ExecMode.SINGLE_NODE);
	}

	@Test
	public void testHP() throws IOException {
		runWriteTest(ExecMode.HYBRID);
	}

	@Test
	public void testSP() throws IOException {
		runWriteTest(ExecMode.SPARK);
	}

	private void runWriteTest(ExecMode platform) throws IOException {
		runWriteTest(platform, 100, 100, 0, 0, 0.0);
	}

	private void runWriteTest(ExecMode platform, int rows, int cols, int min, int max, double sparsity)
		throws IOException {

		ExecMode oldPlatform = rtplatform;
		rtplatform = platform;

		TestConfiguration config = getTestConfiguration(TEST_NAME);
		loadTestConfiguration(config);

		String HOME = SCRIPT_DIR + TEST_DIR;

		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[] {"-explain", "-args", "" + rows, "" + cols, "" + min, "" + max, "" + sparsity,
			output("out.cla"), output("sum.scalar")};

		runTest(null);

		double sumDML = TestUtils.readDMLScalar(output("sum.scalar"));
		MatrixBlock mbr = IOCompressionTestUtils.read(output("out.cla"));
		
		TestUtils.compareScalars(sumDML, mbr.sum(), eps);

		rtplatform = oldPlatform;
	}
}
