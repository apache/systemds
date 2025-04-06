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

package org.apache.sysds.test.functions.pipelines;

import org.apache.sysds.common.Types;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Ignore;
import org.junit.Test;

public class BuiltinFitPipelineTest extends AutomatedTestBase {
	//	private final static String TEST_NAME1 = "prioritized";
	private final static String TEST_NAME1 = "fit_pipelineTest";
	private final static String TEST_CLASS_DIR = SCRIPT_DIR + BuiltinFitPipelineTest.class.getSimpleName() + "/";

	private static final String RESOURCE = SCRIPT_DIR+"functions/pipelines/";
	private static final String DATA_DIR = DATASET_DIR+ "pipelines/";

	private final static String DIRTY = DATA_DIR+ "dirty.csv";
	private final static String META = RESOURCE+ "meta/meta_census.csv";
	private final static String INPUT = RESOURCE+"intermediates/";

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME1,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1,new String[]{"R"}));
	}

	@Ignore
	@Test
	public void testEvalPipClass() {
		evalPip(0.8, "FALSE", INPUT+"/classification/", Types.ExecMode.SINGLE_NODE);
	}

	private void evalPip(double split, String cv, String path, Types.ExecMode et) {
		String HOME = SCRIPT_DIR+"functions/pipelines/";
		Types.ExecMode modeOld = setExecMode(et);

		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME1));
			fullDMLScriptName = HOME + TEST_NAME1 + ".dml";
			programArgs = new String[] {"-stats", "-exec", "singlenode", "-args", DIRTY, META, path, cv,
				String.valueOf(split), output("O")};

			runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);

			//expected loss smaller than default invocation
			Assert.assertTrue(TestUtils.readDMLBoolean(output("O")));
		}
		finally {
			resetExecMode(modeOld);
		}
	}
}
