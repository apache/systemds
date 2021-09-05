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

public class BuiltinTopkCleaningClassificationTest extends AutomatedTestBase {
	private final static String TEST_NAME = "topkcleaningClassificationTest";
	private final static String TEST_CLASS_DIR = SCRIPT_DIR + BuiltinTopkCleaningClassificationTest.class.getSimpleName() + "/";

	private final static String TEST_DIR = "functions/pipelines/";
	private static final String RESOURCE = SCRIPT_DIR+"functions/pipelines/";
	private static final String DATA_DIR = DATASET_DIR+ "pipelines/";

	private final static String DIRTY = DATA_DIR+ "dirty.csv";
	private final static String META = RESOURCE+ "meta/meta_census.csv";
	private final static String OUTPUT = RESOURCE+ "intermediates/classification/";

	private static final String PARAM_DIR = "./scripts/pipelines/properties/";
	private final static String PARAM = PARAM_DIR + "param.csv";
	private final static String PRIMITIVES = PARAM_DIR + "testPrimitives.csv";

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[]{"R"}));
	}

	@Test
	public void testFindBestPipeline1() {
		runtopkCleaning(0.5, 3,5,
			"FALSE", 0,0.8, Types.ExecMode.SINGLE_NODE);
	}

	@Ignore
	public void testFindBestPipeline2() {
		runtopkCleaning(0.1, 3,5,
			"TRUE", 3,0.8,  Types.ExecMode.SINGLE_NODE);
	}

	@Test
	public void testFindBestPipelineHybrid() {
		runtopkCleaning(0.1, 3,5,
			"FALSE", 0,0.8,  Types.ExecMode.HYBRID);
	}

	private void runtopkCleaning(Double sample, int topk, int resources,  String cv, int cvk , double split, Types.ExecMode et) {

		Types.ExecMode modeOld = setExecMode(et);
		String HOME = SCRIPT_DIR + TEST_DIR;
		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] { "-stats", "-exec", "singlenode", "-nvargs", "dirtyData="+DIRTY,
				"metaData="+META, "primitives="+PRIMITIVES, "parameters="+PARAM, "topk="+ topk, "rv="+ resources,
				"sample="+sample, "testCV="+cv, "cvk="+cvk, "split="+split, "output="+OUTPUT, "O="+output("O")};

			runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);

			//expected loss smaller than default invocation
			Assert.assertTrue(TestUtils.readDMLBoolean(output("O")));
		}
		finally {
			resetExecMode(modeOld);
		}
	}

}
