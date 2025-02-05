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

public class BertLayerTest extends AutomatedTestBase{
	private static final String TEST_NAME_FORWARD = "bert_layer_forward";
	private static final String TEST_DIR = "applications/nn/component/";
	private static final String RESOURCE_DIR = "src/test/resources/component/transformers/bert_layer/";

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME_FORWARD, new TestConfiguration(TEST_DIR, TEST_NAME_FORWARD));
	}

	@Test
	public void testBertLayerForwardNormalTanh() {
		runBertLayerTest("test1", 5, 4, 6, 2, 3, 7, "tanh", 0, TEST_NAME_FORWARD, 
            1e-5, true);
	}

	@Test
	public void testBertLayerForwardNormalGelu() {
		runBertLayerTest("test2", 4, 4, 8, 2, 4, 7, "gelu", 0, TEST_NAME_FORWARD, 
            1e-5, true);
	}

	private void runBertLayerTest(String testSuffix, int batchSize, int seqLength, int embeddingDim, int numHeads,
            int perHeadEmbeddingDim, int intermediateEmbeddingDim, String activation, int debug, String testname, double precision, 
            boolean isForward) {
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
					String.valueOf(debug), String.valueOf(batchSize), 
                    String.valueOf(seqLength), String.valueOf(embeddingDim),
					String.valueOf(numHeads), String.valueOf(perHeadEmbeddingDim),
                    String.valueOf(intermediateEmbeddingDim), activation,
					RESOURCE_DIR + "input_states_" + testSuffix + ".csv",
					RESOURCE_DIR + "input_W_Q_" + testSuffix + ".csv",
					RESOURCE_DIR + "input_b_Q_" + testSuffix + ".csv",
                    RESOURCE_DIR + "input_W_K_" + testSuffix + ".csv",
					RESOURCE_DIR + "input_b_K_" + testSuffix + ".csv",
                    RESOURCE_DIR + "input_W_V_" + testSuffix + ".csv",
					RESOURCE_DIR + "input_b_V_" + testSuffix + ".csv",
                    RESOURCE_DIR + "input_W_context_" + testSuffix + ".csv",
					RESOURCE_DIR + "input_b_context_" + testSuffix + ".csv",
                    RESOURCE_DIR + "input_W_intermediate_" + testSuffix + ".csv",
					RESOURCE_DIR + "input_b_intermediate_" + testSuffix + ".csv",
                    RESOURCE_DIR + "input_W_out_" + testSuffix + ".csv",
					RESOURCE_DIR + "input_b_out_" + testSuffix + ".csv",
                    RESOURCE_DIR + "input_gamma_ln1_" + testSuffix + ".csv",
					RESOURCE_DIR + "input_beta_ln1_" + testSuffix + ".csv",
                    RESOURCE_DIR + "input_gamma_ln2_" + testSuffix + ".csv",
					RESOURCE_DIR + "input_beta_ln2_" + testSuffix + ".csv",
					RESOURCE_DIR + "output_states_" + testSuffix + ".csv",
					RESOURCE_DIR + "output_attention_" + testSuffix + ".csv",
					output("states_error"),
					output("attention_error"), 
				};
			}

			// Run the test
			runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);

			// Compare results
			if (isForward) {
				double statesMaxError = (Double) readDMLScalarFromOutputDir("states_error").values().toArray()[0];
				assert statesMaxError < precision;
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
