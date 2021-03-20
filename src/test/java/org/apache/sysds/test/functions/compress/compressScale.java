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

package org.apache.sysds.test.functions.compress;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types;
import org.apache.sysds.lops.LopProperties;
import org.apache.sysds.lops.LopProperties.ExecType;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.DMLCompressionStatistics;
import org.junit.Test;

public class compressScale extends AutomatedTestBase {
	private static final Log LOG = LogFactory.getLog(compressScale.class.getName());

	protected String getTestClassDir() {
		return getTestDir() + this.getClass().getSimpleName() + "/";
	}

	protected String getTestName() {
		return "scale";
	}

	protected String getTestDir() {
		return "functions/compress/compressScale/";
	}

	// @Test
	// public void testInstruction_01() {
	//	 compressTest(4, 1000000, 0.2, ExecType.CP, -100, 1000, 1, 1);
	// }

	@Test
	public void testInstruction_01_1() {
		compressTest(50, 1000000, 0.2, ExecType.CP, 1, 2, 1, 1);
	}

	// @Test
	// public void testInstruction_02() {
	// compressTest(10, 200000, 0.2, ExecType.CP, 0, 5, 0, 1);
	// }

	// @Test
	// public void testInstruction_03() {
	// compressTest(10, 200000, 0.2, ExecType.CP, 0, 5, 0, 0);
	// }

	// @Test
	// public void testInstruction_04() {
	// compressTest(10, 200000, 0.2, ExecType.CP, 0, 5, 1, 0);
	// }

	public void compressTest(int cols, int rows, double sparsity, LopProperties.ExecType instType, int min, int max,
		int scale, int center) {

		Types.ExecMode platformOld = setExecMode(instType);
		try {

			fullDMLScriptName = SCRIPT_DIR + "/" + getTestDir() + getTestName() + ".dml";
			loadTestConfiguration(getTestConfiguration(getTestName()));

			// Default arguments
			programArgs = new String[] {"-config", "", "-nvargs", "cols=" + cols, "rows=" + rows,
				"sparsity=" + sparsity, "min=" + min, "max= " + max, "scale=" + scale, "center=" + center};

			// Default execution
			programArgs[1] = configPath("SystemDS-config-default.xml");
			double outStd = Double.parseDouble(runTest(null).toString().split("\n")[0].split(" ")[0]);
			LOG.debug("ULA : " + outStd);

			programArgs[1] = configPath("SystemDS-config-compress-cost-RLE.xml");
			double RLEoutC = Double.parseDouble(runTest(null).toString().split("\n")[0].split(" ")[0]);
			assertTrue(DMLCompressionStatistics.haveCompressed());
			DMLCompressionStatistics.reset();
			LOG.debug("RLE : " + RLEoutC);
			
			programArgs[1] = configPath("SystemDS-config-compress-cost-OLE.xml");
			double OLEOutC = Double.parseDouble(runTest(null).toString().split("\n")[0].split(" ")[0]);
			assertTrue(DMLCompressionStatistics.haveCompressed());
			DMLCompressionStatistics.reset();
			LOG.debug("OLE : " + OLEOutC);
			
			programArgs[1] = configPath("SystemDS-config-compress-cost-DDC.xml");
			double DDCoutC = Double.parseDouble(runTest(null).toString().split("\n")[0].split(" ")[0]);
			assertTrue(DMLCompressionStatistics.haveCompressed());
			DMLCompressionStatistics.reset();
			LOG.debug("DDC : " + DDCoutC);
			
			programArgs[1] = configPath("SystemDS-config-compress-cost.xml");
			double ALLoutC = Double.parseDouble(runTest(null).toString().split("\n")[0].split(" ")[0]);
			assertTrue(DMLCompressionStatistics.haveCompressed());
			DMLCompressionStatistics.reset();
			LOG.debug("CLA : " + ALLoutC);

			assertEquals(outStd, OLEOutC, 0.1);
			assertEquals(outStd, RLEoutC, 0.1);
			assertEquals(outStd, DDCoutC, 0.1);
			assertEquals(outStd, ALLoutC, 0.1);

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
		disableConfigFile = true;
		TestUtils.clearAssertionInformation();
		addTestConfiguration(getTestName(), new TestConfiguration(getTestClassDir(), getTestName()));
	}

	private String configPath(String file) {
		String out = (SCRIPT_DIR + getTestDir() + file).substring(2);
		return out;
	}
}
