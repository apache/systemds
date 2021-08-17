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

public class BuiltinTopkCleaningRegressionTest extends AutomatedTestBase{
	private final static String TEST_NAME1 = "topkcleaningRegressionTest";
	private final static String TEST_CLASS_DIR = SCRIPT_DIR + BuiltinTopkCleaningRegressionTest.class.getSimpleName() + "/";

	private static final String RESOURCE = SCRIPT_DIR+"functions/pipelines/";
	//	private static final String DATA_DIR = DATASET_DIR+ "pipelines/";

	private final static String DIRTY = DATASET_DIR+ "Salaries.csv";
	private final static String OUTPUT = RESOURCE+"intermediates/";
	private static final String PARAM_DIR = "./scripts/pipelines/properties/";
	private final static String PARAM = PARAM_DIR + "param.csv";
	private final static String PRIMITIVES = PARAM_DIR + "primitives.csv";

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME1,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1,new String[]{"R"}));
	}

	@Test
	public void testRegressionPipelinesCP() {
		runFindPipelineTest(1.0, 5,20, 10,
			"lm", Types.ExecMode.SINGLE_NODE);
	}

	@Ignore
	public void testRegressionPipelinesHybrid() {
		runFindPipelineTest(1.0, 5,5, 2,
			"lm", Types.ExecMode.HYBRID);
	}


	private void runFindPipelineTest(Double sample, int topk, int resources, int crossfold,
		String target, Types.ExecMode et) {

		//		setOutputBuffering(true);
		String HOME = SCRIPT_DIR+"functions/pipelines/" ;
		Types.ExecMode modeOld = setExecMode(et);
		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME1));
			fullDMLScriptName = HOME + TEST_NAME1 + ".dml";
			programArgs = new String[] {"-stats", "-exec", "singlenode", "-nvargs", "dirtyData="+DIRTY,
				"primitives="+PRIMITIVES, "parameters="+PARAM, "sampleSize="+ sample, "topk="+ topk,
				"rv="+ resources, "sample="+ sample, "output="+OUTPUT, "target="+target, "O="+output("O")};

			runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);

			//expected loss smaller than default invocation
			Assert.assertTrue(TestUtils.readDMLBoolean(output("O")));
		}
		finally {
			resetExecMode(modeOld);
		}
	}

}
