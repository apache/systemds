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

package org.apache.sysds.test.functions.compress.instructionsSpark;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public abstract class CompressedSparkInstructionsTest extends AutomatedTestBase {

	final double eps = 0.00000001;

	protected String getTestClassDir() {
		return getTestDir() + this.getClass().getSimpleName() + "/";
	}

	protected String getTestName() {
		return "compress";
	}

	protected String getTestDir() {
		return "functions/compress/instructionsSpark/";
	}

	public abstract double getDensity();

	@Test
	public void testSum() {
		run("sum");
	}

	@Test
	public void testMatrixMult() {
		run("mm");
	}

	private void run(String name) {
		run(10, 2500, getDensity(), 1, 5, name);
	}

	private void run(int cols, int rows, double sparsity, int min, int max, String name) {

		OptimizerUtils.ALLOW_SCRIPT_LEVEL_COMPRESS_COMMAND = true;
		Types.ExecMode platformOld = setExecMode(ExecType.SPARK);
		try {
			setOutputBuffering(true);
			loadTestConfiguration(getTestConfiguration(getTestName()));
			fullDMLScriptName = SCRIPT_DIR + "/" + getTestDir() + "compress_" + name + ".dml";
			programArgs = new String[] {"-explain", "-stats", "-args", s(rows), s(cols), s(min), s(max), s(sparsity)};
			String stdout = runTest(null).toString();
			parseOutAndVerify(stdout);
		}
		catch(Exception e) {
			e.printStackTrace();
			assertTrue("Exception in execution: " + e.getMessage(), false);
		}
		finally {
			rtplatform = platformOld;
		}
	}

	private void parseOutAndVerify(String out) {
		double C = 0;
		double uC = 0;
		try {

			String[] lines = out.split("\n");
			for(int i = 0; i < lines.length; i++)
				if(lines[i].equals("RESULTS:")) {
					C = Double.parseDouble(lines[i + 1].split("=")[1]);
					uC = Double.parseDouble(lines[i + 2].split("=")[1]);
					break;
				}
		}
		catch(Exception e) {
			fail("Error when passing results:\n" + out);
		}
		assertEquals(out, C, uC, eps);
	}

	private String s(int i) {
		return String.valueOf(i);
	}

	private String s(double i) {
		return String.valueOf(i);
	}

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(getTestName(), new TestConfiguration(getTestClassDir(), getTestName()));
	}

	// /**
	// * Override default configuration with custom test configuration to ensure scratch space and local temporary
	// * directory locations are also updated.
	// */
	// @Override
	// protected File getConfigTemplateFile() {
	// return TEST_CONF_FILE;
	// }
}
