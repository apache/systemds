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
	private static final String TEST_NAME_BACKWARD = "bert_layer_backward";
	private static final String TEST_DIR = "applications/nn/component/";
	private static final String RESOURCE_DIR = "src/test/resources/component/transformers/bert_layer/";

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME_FORWARD, new TestConfiguration(TEST_DIR, TEST_NAME_FORWARD));
		addTestConfiguration(TEST_NAME_BACKWARD, new TestConfiguration(TEST_DIR, TEST_NAME_BACKWARD));
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

	@Test
	public void testBertLayerBackwardNormalGelu() {
		runBertLayerTest("test3", 2, 3, 8, 2, 4, 5, "gelu", 0, TEST_NAME_BACKWARD, 
            1e-4, false);
	}

	@Test
	public void testBertLayerBackwardSameDimsTanh() {
		runBertLayerTest("test4", 4, 4, 4, 2, 2, 4, "tanh", 0, TEST_NAME_BACKWARD, 
            1e-4, false);
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
			} else {
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
					RESOURCE_DIR + "input_dstates_" + testSuffix + ".csv",
					RESOURCE_DIR + "output_dstates_" + testSuffix + ".csv",
					RESOURCE_DIR + "output_dW_Q_" + testSuffix + ".csv",
					RESOURCE_DIR + "output_db_Q_" + testSuffix + ".csv",
                    RESOURCE_DIR + "output_dW_K_" + testSuffix + ".csv",
					RESOURCE_DIR + "output_db_K_" + testSuffix + ".csv",
                    RESOURCE_DIR + "output_dW_V_" + testSuffix + ".csv",
					RESOURCE_DIR + "output_db_V_" + testSuffix + ".csv",
                    RESOURCE_DIR + "output_dW_context_" + testSuffix + ".csv",
					RESOURCE_DIR + "output_db_context_" + testSuffix + ".csv",
                    RESOURCE_DIR + "output_dW_intermediate_" + testSuffix + ".csv",
					RESOURCE_DIR + "output_db_intermediate_" + testSuffix + ".csv",
                    RESOURCE_DIR + "output_dW_out_" + testSuffix + ".csv",
					RESOURCE_DIR + "output_db_out_" + testSuffix + ".csv",
                    RESOURCE_DIR + "output_dgamma_ln1_" + testSuffix + ".csv",
					RESOURCE_DIR + "output_dbeta_ln1_" + testSuffix + ".csv",
                    RESOURCE_DIR + "output_dgamma_ln2_" + testSuffix + ".csv",
					RESOURCE_DIR + "output_dbeta_ln2_" + testSuffix + ".csv",
					output("din_error"),
					output("dW_Q_error"),
					output("db_Q_error"),
					output("dW_K_error"),
					output("db_K_error"),
					output("dW_V_error"),
					output("db_V_error"),
					output("dW_context_error"),
					output("db_context_error"),
					output("dW_intermediate_error"),
					output("db_intermediate_error"),
					output("dW_out_error"),
					output("db_out_error"),
					output("dgamma_ln1_error"),
					output("dbeta_ln1_error"),
					output("dgamma_ln2_error"),
					output("dbeta_ln2_error"),
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
				double dinMaxError = (Double) readDMLScalarFromOutputDir("din_error").values().toArray()[0];
				assert dinMaxError < precision;
				double dWQMaxError = (Double) readDMLScalarFromOutputDir("dW_Q_error").values().toArray()[0];
				assert dWQMaxError < precision;
				double dbQMaxError = (Double) readDMLScalarFromOutputDir("db_Q_error").values().toArray()[0];
				assert dbQMaxError < precision;
				double dWKMaxError = (Double) readDMLScalarFromOutputDir("dW_K_error").values().toArray()[0];
				assert dWKMaxError < precision;
				double dbKMaxError = (Double) readDMLScalarFromOutputDir("db_K_error").values().toArray()[0];
				assert dbKMaxError < precision;
				double dWVMaxError = (Double) readDMLScalarFromOutputDir("dW_V_error").values().toArray()[0];
				assert dWVMaxError < precision;
				double dbVMaxError = (Double) readDMLScalarFromOutputDir("db_V_error").values().toArray()[0];
				assert dbVMaxError < precision;
				double dWContextMaxError = (Double) readDMLScalarFromOutputDir("dW_context_error").values().toArray()[0];
				assert dWContextMaxError < precision;
				double dbContextMaxError = (Double) readDMLScalarFromOutputDir("db_context_error").values().toArray()[0];
				assert dbContextMaxError < precision;
				double dWIntermediateMaxError = (Double) readDMLScalarFromOutputDir("dW_intermediate_error").values().toArray()[0];
				assert dWIntermediateMaxError < precision;
				double dbIntermediateMaxError = (Double) readDMLScalarFromOutputDir("db_intermediate_error").values().toArray()[0];
				assert dbIntermediateMaxError < precision;
				double dWOutMaxError = (Double) readDMLScalarFromOutputDir("dW_out_error").values().toArray()[0];
				assert dWOutMaxError < precision;
				double dbOutMaxError = (Double) readDMLScalarFromOutputDir("db_out_error").values().toArray()[0];
				assert dbOutMaxError < precision;
				double dgammaLn1MaxError = (Double) readDMLScalarFromOutputDir("dgamma_ln1_error").values().toArray()[0];
				assert dgammaLn1MaxError < precision;
				double dbetaLn1MaxError = (Double) readDMLScalarFromOutputDir("dbeta_ln1_error").values().toArray()[0];
				assert dbetaLn1MaxError < precision;
				double dgammaLn2MaxError = (Double) readDMLScalarFromOutputDir("dgamma_ln2_error").values().toArray()[0];
				assert dgammaLn2MaxError < precision;
				double dbetaLn2MaxError = (Double) readDMLScalarFromOutputDir("dbeta_ln2_error").values().toArray()[0];
				assert dbetaLn2MaxError < precision;
			}
		} catch (Throwable ex) {
			ex.printStackTrace(System.out); // Log or debug all exceptions or errors
			throw new RuntimeException(ex);
		} finally {
			resetExecMode(platformOld);
		}
	}
}
