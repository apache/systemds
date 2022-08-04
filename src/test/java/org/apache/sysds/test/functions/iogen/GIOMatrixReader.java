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

import java.io.File;

public abstract class GIOMatrixReader extends AutomatedTestBase {

	protected final static String TEST_DIR = "functions/iogen/";
	protected final static String TEST_CLASS_DIR = TEST_DIR + GIOMatrixReader.class.getSimpleName() + "/";

	protected abstract int getId();

	protected String getInputDatasetFileName() {
		return "dataset_" + getId() + ".dat";
	}

	protected String getInputSampleMatrixFileName() {
		return "sampleMatrix_" + getId() + ".mtx";
	}

	protected String getInputSampleRawFileName() {
		return "sampleMatrix_" + getId() + ".raw";
	}

	protected String getOutputGIO() {
		return "GIO" + getTestName()+"_"+ getId()+".java";
	}

	@Test
	public void testSequential_CP1() {
		runGIOTest(getId(), false);
	}

	protected abstract String getTestName();

	@Override public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(getTestName(), new TestConfiguration(TEST_DIR, getTestName(), new String[] {"Y"}));
	}

	@SuppressWarnings("unused") protected void runGIOTest(int testNumber, boolean parallel) {

		Types.ExecMode oldPlatform = rtplatform;
		rtplatform = Types.ExecMode.SINGLE_NODE;

		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		boolean oldpar = CompilerConfig.FLAG_PARREADWRITE_TEXT;

		try {
			CompilerConfig.FLAG_DYN_RECOMPILE = false;
			CompilerConfig.FLAG_PARREADWRITE_TEXT = false;

			TestConfiguration config = getTestConfiguration(getTestName());
			loadTestConfiguration(config);
			setOutputBuffering(true);
			setOutAndExpectedDeletionDisabled(true);

			String HOME = SCRIPT_DIR + TEST_DIR;
			String inputDataset = HOME + INPUT_DIR + getInputDatasetFileName();
			String inputSampleMatrix = HOME + INPUT_DIR + getInputSampleMatrixFileName();
			String inputSampleRaw = HOME + INPUT_DIR + getInputSampleRawFileName();
			String outputSrc = HOME +"iogensrc/" + getOutputGIO();
			String outputMatrix = output(getInputDatasetFileName());

			File outDir = new File(HOME + OUTPUT_DIR);
			if(!outDir.exists())
				outDir.mkdirs();

			outDir = new File(HOME +"iogensrc/");
			if(!outDir.exists())
				outDir.mkdirs();


			fullDMLScriptName = HOME + getTestName() + "_" + testNumber + ".dml";
			programArgs = new String[] {"-args", inputDataset, inputSampleMatrix, inputSampleRaw, outputSrc, outputMatrix };

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
}
