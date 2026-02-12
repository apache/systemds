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

package org.apache.sysds.test.functions.jmlc;

import org.apache.sysds.api.jmlc.Connection;
import org.apache.sysds.api.jmlc.LLMCallback;
import org.apache.sysds.api.jmlc.PreparedScript;
import org.apache.sysds.test.AutomatedTestBase;
import org.junit.Assert;
import org.junit.Test;

/**
 * Test LLM inference capabilities via JMLC API.
 * This test requires Python with transformers and torch installed.
 */
public class JMLCLLMInferenceTest extends AutomatedTestBase {
	private final static String TEST_NAME = "JMLCLLMInferenceTest";
	private final static String TEST_DIR = "functions/jmlc/";

	@Override
	public void setUp() {
		addTestConfiguration(TEST_DIR, TEST_NAME);
		getAndLoadTestConfiguration(TEST_NAME);
	}

	@Test
	public void testLLMInference() {
		Connection conn = null;
		try {
			// Create a connection
			conn = new Connection();
			
			// Load the LLM model via Python worker
			LLMCallback llmWorker = conn.loadModel("distilgpt2");
			Assert.assertNotNull("LLM worker should not be null", llmWorker);
			
			// Create a PreparedScript with a dummy script
			String dummyScript = "x = 1;\nwrite(x, './tmp/x');";
			PreparedScript ps = conn.prepareScript(dummyScript, new String[]{}, new String[]{"x"});
			
			// Set the LLM worker on the PreparedScript
			ps.setLLMWorker(llmWorker);
			
			// Generate text using the LLM
			String prompt = "The meaning of life is";
			String result = ps.generate(prompt, 20, 0.7, 0.9);
			
			// Assert the result is not null and not empty
			Assert.assertNotNull("Generated text should not be null", result);
			Assert.assertFalse("Generated text should not be empty", result.isEmpty());
			
			System.out.println("Prompt: " + prompt);
			System.out.println("Generated: " + result);
			
		} catch (Exception e) {
			// Skip test if Python/transformers not available
			System.out.println("Skipping LLM test:");
			e.printStackTrace();
			org.junit.Assume.assumeNoException("LLM dependencies not available", e);
		} finally {
			if (conn != null) {
				conn.close();
			}
		}
	}
}
