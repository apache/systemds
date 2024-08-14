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

package org.apache.sysds.test.functions.compress.reshape;

import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public class CompressedReshapeTest extends AutomatedTestBase {
	protected static final Log LOG = LogFactory.getLog(CompressedReshapeTest.class.getName());

	private final static String TEST_DIR = "functions/compress/reshape/";

	protected String getTestClassDir() {
		return getTestDir() + this.getClass().getSimpleName() + "/";
	}

	protected String getTestName() {
		return "reshape1";
	}

	protected String getTestDir() {
		return TEST_DIR;
	}

	@Test
	public void testReshape_01_1to2_sparse() {
		reshapeTest(1, 1000, 2, 500, 0.2, ExecType.CP, 0, 5, "01");
	}

	@Test
	public void testReshape_01_2to4_sparse() {
		reshapeTest(2, 500, 4, 250, 0.2, ExecType.CP, 0, 5, "01");
	}

	@Test
	public void testReshape_01_1to10_sparse() {
		reshapeTest(1, 10000, 10, 1000, 0.2, ExecType.CP, 0, 5, "01");
	}

	@Test
	public void testReshape_01_1to2_dense() {
		reshapeTest(1, 1000, 2, 500, 1.0, ExecType.CP, 0, 5, "01");
	}

	@Test
	public void testReshape_01_2to4_dense() {
		reshapeTest(2, 500, 4, 250, 1.0, ExecType.CP, 0, 5, "01");
	}

	@Test
	public void testReshape_01_1to10_dense() {
		reshapeTest(1, 10000, 10, 1000, 1.0, ExecType.CP, 0, 5, "01");
	}

	@Test
	public void testReshape_02_1to2_sparse() {
		reshapeTest(1, 1000, 2, 500, 0.2, ExecType.CP, 0, 10, "02");
	}

	@Test
	public void testReshape_02_1to2_dense() {
		reshapeTest(1, 1000, 2, 500, 1.0, ExecType.CP, 0, 10, "02");
	}

	@Test
	public void testReshape_03_1to2_sparse() {
		reshapeTest(1, 1000, 2, 500, 0.2, ExecType.CP, 0, 10, "03");
	}

	@Test
	public void testReshape_03_1to2_dense() {
		reshapeTest(1, 1000, 2, 500, 1.0, ExecType.CP, 0, 10, "03");
	}

	public void reshapeTest(int cols, int rows, int reCol, int reRows, double sparsity, ExecType instType, int min,
		int max, String name) {

		OptimizerUtils.ALLOW_SCRIPT_LEVEL_COMPRESS_COMMAND = true;
		Types.ExecMode platformOld = setExecMode(instType);

		CompressedMatrixBlock.debug = true;
		CompressedMatrixBlock.allowCachingUncompressed = false;
		try {

			super.setOutputBuffering(true);
			loadTestConfiguration(getTestConfiguration(getTestName()));

			fullDMLScriptName = SCRIPT_DIR + "/" + getTestClassDir() + name + ".dml";

			programArgs = new String[] {"-stats", "100", "-nvargs", "cols=" + cols, "rows=" + rows, "reCols=" + reCol,
				"reRows=" + reRows, "sparsity=" + sparsity, "min=" + min, "max= " + max};
			String s = runTest(null).toString();

			if(s.contains("Failed"))
				fail(s);
			else
				LOG.debug(s);

		}
		catch(Exception e) {
			e.printStackTrace();
			assertTrue("Exception in execution: " + e.getMessage(), false);
		}
		finally {
			rtplatform = platformOld;
		}
	}

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(getTestName(), new TestConfiguration(getTestClassDir(), getTestName()));
	}

}
