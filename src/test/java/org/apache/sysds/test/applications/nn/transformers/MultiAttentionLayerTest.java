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
package org.apache.sysds.test.applications.nn.transformers;

import org.apache.sysds.common.Types;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public class MultiAttentionLayerTest extends AutomatedTestBase {
	private static final String TEST_NAME_FORWARD = "multi_attention_forward";
	private static final String TEST_NAME_BACKWARD = "multi_attention_backward";
	private static final String TEST_DIR = "applications/nn/component/";
	private static final String RESOURCE_DIR = "src/test/resources/component/transformers/multi_attention_layer/";

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME_FORWARD, new TestConfiguration(TEST_DIR, TEST_NAME_FORWARD));
		addTestConfiguration(TEST_NAME_BACKWARD, new TestConfiguration(TEST_DIR, TEST_NAME_BACKWARD));
	}

	@Test
	public void testMultiAttentionForwardSimple() {
		runMultiAttentionTest("test1", 2, 3, 4, 5, 0, TEST_NAME_FORWARD, 1e-5, true);
	}

	@Test
	public void testMultiAttentionForwardLarge() {
		runMultiAttentionTest("test2", 8, 12, 10, 4, 0, TEST_NAME_FORWARD, 1e-5, true);
	}

	@Test
	public void testMultiAttentionForwardSmall() {
		runMultiAttentionTest("test3", 1, 1, 1, 1, 0, TEST_NAME_FORWARD, 1e-5, true);
	}

	@Test
	public void testMultiAttentionBackwardSimple() {
		runMultiAttentionTest("test4", 2, 3, 4, 5, 0, TEST_NAME_BACKWARD, 1e-5, false);
	}

	@Test
	public void testMultiAttentionBackwardLarge() {
		runMultiAttentionTest("test5", 8, 12, 10, 5, 0, TEST_NAME_BACKWARD, 1e-5, false);
	}

	@Test
	public void testMultiAttentionBackwardSmall() {
		runMultiAttentionTest("test6", 1, 1, 1, 1, 0, TEST_NAME_BACKWARD, 1e-5, false);
	}

	private void runMultiAttentionTest(String testSuffix, int batchSize, int seqLength, int numHeads, int embeddingDim,
			int debug, String testname, double precision, boolean isForward) {
		// Set execution platform
		Types.ExecMode platformOld = setExecMode(Types.ExecMode.SINGLE_NODE);

		try {
			// Load test configuration
			getAndLoadTestConfiguration(testname);
			fullDMLScriptName = getScript();

			// Program arguments
			if (isForward) {
				programArgs = new String[] { 
					"-stats", "-args",
					String.valueOf(batchSize), String.valueOf(seqLength),
					String.valueOf(numHeads), String.valueOf(embeddingDim),
					String.valueOf(debug),
					RESOURCE_DIR + "input_query_" + testSuffix + ".csv",
					RESOURCE_DIR + "input_key_" + testSuffix + ".csv",
					RESOURCE_DIR + "input_value_" + testSuffix + ".csv",
					RESOURCE_DIR + "output_context_" + testSuffix + ".csv",
					RESOURCE_DIR + "output_attention_" + testSuffix + ".csv",
					output("context_error"),
					output("attention_error"), 
				};
			} else {
				programArgs = new String[] { 
					"-stats", "-args",
					String.valueOf(batchSize), String.valueOf(seqLength),
					String.valueOf(numHeads), String.valueOf(embeddingDim),
					String.valueOf(debug),
					RESOURCE_DIR + "input_query_" + testSuffix + ".csv",
					RESOURCE_DIR + "input_key_" + testSuffix + ".csv",
					RESOURCE_DIR + "input_value_" + testSuffix + ".csv",
					RESOURCE_DIR + "input_dcontext_" + testSuffix + ".csv",
					RESOURCE_DIR + "input_attention_" + testSuffix + ".csv",
					RESOURCE_DIR + "output_dquery_" + testSuffix + ".csv",
					RESOURCE_DIR + "output_dkey_" + testSuffix + ".csv",
					RESOURCE_DIR + "output_dvalue_" + testSuffix + ".csv",
					output("dquery_error"),
					output("dkey_error"), 
					output("dvalue_error"), 
				};
			}

			// Run the test
			runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);

			// Compare results
			if (isForward) {
				double contextMaxError = (Double) readDMLScalarFromOutputDir("context_error").values().toArray()[0];
				assert contextMaxError < precision;
				double attentionMaxError = (Double) readDMLScalarFromOutputDir("attention_error").values().toArray()[0];
				assert attentionMaxError < precision;
			} else {
				double dqueryMaxError = (Double) readDMLScalarFromOutputDir("dquery_error").values().toArray()[0];
				assert dqueryMaxError < precision;
				double dkeyMaxError = (Double) readDMLScalarFromOutputDir("dkey_error").values().toArray()[0];
				assert dkeyMaxError < precision;
				double dvalueMaxError = (Double) readDMLScalarFromOutputDir("dvalue_error").values().toArray()[0];
				assert dvalueMaxError < precision;
			}
		} catch (Throwable ex) {
			ex.printStackTrace(System.out); // Log or debug all exceptions or errors
			throw new RuntimeException(ex);
		} finally {
			resetExecMode(platformOld);
		}
	}
}
