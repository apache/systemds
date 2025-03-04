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

package org.apache.sysds.test.functions.compress.table;

import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.io.File;

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

public class CompressedTableOverwriteTest extends AutomatedTestBase {
	protected static final Log LOG = LogFactory.getLog(CompressedTableOverwriteTest.class.getName());

	private final static String TEST_DIR = "functions/compress/table/";

	protected String getTestClassDir() {
		return getTestDir() + this.getClass().getSimpleName() + "/";
	}

	protected String getTestName() {
		return "table";
	}

	protected String getTestDir() {
		return TEST_DIR;
	}

	@Test
	public void testRewireTable_2() {
		rewireTableTest(10, 2, 0.2, ExecType.CP, "01");
	}

	@Test
	public void testRewireTable_20() {
		rewireTableTest(30, 20, 0.2, ExecType.CP, "01");
	}

	@Test
	public void testRewireTable_80() {
		rewireTableTest(100, 80, 0.2, ExecType.CP, "01");
	}

	@Test
	public void testRewireTable_80_1000() {
		rewireTableTest(1000, 80, 0.2, ExecType.CP, "01");
	}

	@Test
	public void testRewireTable_80_1000_dense() {
		rewireTableTest(1000, 80, 1.0, ExecType.CP, "01");
	}


	public void rewireTableTest(int rows, int unique, double sparsity, ExecType instType, String name) {

		OptimizerUtils.ALLOW_SCRIPT_LEVEL_COMPRESS_COMMAND = true;
		Types.ExecMode platformOld = setExecMode(instType);

		CompressedMatrixBlock.debug = true;
		CompressedMatrixBlock.allowCachingUncompressed = false;
		try {

			super.setOutputBuffering(true);
			loadTestConfiguration(getTestConfiguration(getTestName()));
			fullDMLScriptName = SCRIPT_DIR + "/" + getTestClassDir() + name + ".dml";
			programArgs = new String[] {"-stats", "100", "-nvargs", "rows=" + rows, "unique=" + unique,
				"sparsity=" + sparsity};
			String s = runTest(null).toString();

			if(s.contains("Failed"))
				fail(s);
			// else
				// LOG.debug(s);

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

	@Override
	protected File getConfigTemplateFile() {
		return new File("./src/test/scripts/functions/compress/SystemDS-config-compress.xml");
	}
}
