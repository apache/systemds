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

package org.apache.sysds.test.functions.frame;

import java.util.HashMap;

import static org.junit.Assert.assertEquals;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Ignore;
import org.junit.Test;

public class FrameReplaceTest extends AutomatedTestBase {
	// private static final Log LOG = LogFactory.getLog(FrameReplaceTest.class.getName());
	private final static String TEST_DIR = "functions/frame/";
	private final static String TEST_NAME = "ReplaceTest";
	private final static String TEST_CLASS_DIR = TEST_DIR + FrameReplaceTest.class.getSimpleName() + "/";

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"S.scalar"}));
	}

	@Test
	public void testParforFrameIntermediatesCP() {
		runReplaceTest(ExecType.CP);
	}

	@Test
	@Ignore
	public void testParforFrameIntermediatesSpark() {
		runReplaceTest(ExecType.SPARK);
	}

	private void runReplaceTest(ExecType et) {
		ExecMode platformOld = rtplatform;
		switch(et) {
			case SPARK:
				rtplatform = ExecMode.SPARK;
				break;
			default:
				rtplatform = ExecMode.HYBRID;
				break;
		}

		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if(rtplatform == ExecMode.SPARK || rtplatform == ExecMode.HYBRID)
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		setOutputBuffering(true);
		try {
			// setup testcase
			getAndLoadTestConfiguration(TEST_NAME);
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-nvargs", "out_S=" + output("S")};

			// run test
			runTest(null);
			HashMap<MatrixValue.CellIndex, Double> val = readDMLScalarFromOutputDir("S");
			assertEquals(1.0, val.get(new MatrixValue.CellIndex(1, 1)), 0.0);

		}
		catch(Exception ex) {
			throw new RuntimeException(ex);
		}
		finally {
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}

}
