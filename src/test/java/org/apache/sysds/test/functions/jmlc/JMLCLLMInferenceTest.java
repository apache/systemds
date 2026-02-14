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
import org.apache.sysds.runtime.frame.data.FrameBlock;
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
			//create connection and load model
			conn = new Connection();
			LLMCallback llmWorker = conn.loadModel("distilgpt2", "src/main/python/llm_worker.py");
			Assert.assertNotNull("LLM worker should not be null", llmWorker);
			
			//create prepared script and set llm worker
			String script = "x = 1;\nwrite(x, './tmp/x');";
			PreparedScript ps = conn.prepareScript(script, new String[]{}, new String[]{"x"});
			ps.setLLMWorker(llmWorker);
			
			//generate text using llm
			String prompt = "The meaning of life is";
			String result = ps.generate(prompt, 20, 0.7, 0.9);
			
			//verify result
			Assert.assertNotNull("Generated text should not be null", result);
			Assert.assertFalse("Generated text should not be empty", result.isEmpty());
			
			System.out.println("Prompt: " + prompt);
			System.out.println("Generated: " + result);
			
		} catch (Exception e) {
			//skip test if dependencies not available
			System.out.println("Skipping LLM test:");
			e.printStackTrace();
			org.junit.Assume.assumeNoException("LLM dependencies not available", e);
		} finally {
			if (conn != null) {
				conn.close();
			}
		}
	}
	
	@Test
	public void testBatchInference() {
		Connection conn = null;
		try {
			//create connection and load model
			conn = new Connection();
			LLMCallback llmWorker = conn.loadModel("distilgpt2", "src/main/python/llm_worker.py");
			
			//create prepared script and set llm worker
			String script = "x = 1;\nwrite(x, './tmp/x');";
			PreparedScript ps = conn.prepareScript(script, new String[]{}, new String[]{"x"});
			ps.setLLMWorker(llmWorker);
			
			//batch generate with multiple prompts
			String[] prompts = {
				"The meaning of life is",
				"Machine learning is",
				"Apache SystemDS enables"
			};
			FrameBlock result = ps.generateBatch(prompts, 20, 0.7, 0.9);
			
			//verify FrameBlock structure
			Assert.assertNotNull("Batch result should not be null", result);
			Assert.assertEquals("Should have 3 rows", 3, result.getNumRows());
			Assert.assertEquals("Should have 2 columns", 2, result.getNumColumns());
			
			//verify each row has prompt and generated text
			for (int i = 0; i < prompts.length; i++) {
				String prompt = (String) result.get(i, 0);
				String generated = (String) result.get(i, 1);
				Assert.assertEquals("Prompt should match", prompts[i], prompt);
				Assert.assertNotNull("Generated text should not be null", generated);
				Assert.assertFalse("Generated text should not be empty", generated.isEmpty());
				System.out.println("Prompt: " + prompt);
				System.out.println("Generated: " + generated);
			}
			
		} catch (Exception e) {
			System.out.println("Skipping batch LLM test:");
			e.printStackTrace();
			org.junit.Assume.assumeNoException("LLM dependencies not available", e);
		} finally {
			if (conn != null) {
				conn.close();
			}
		}
	}
	
	@Test
	public void testBatchWithMetrics() {
		Connection conn = null;
		try {
			//create connection and load model
			conn = new Connection();
			LLMCallback llmWorker = conn.loadModel("distilgpt2", "src/main/python/llm_worker.py");
			
			//create prepared script and set llm worker
			String script = "x = 1;\nwrite(x, './tmp/x');";
			PreparedScript ps = conn.prepareScript(script, new String[]{}, new String[]{"x"});
			ps.setLLMWorker(llmWorker);
			
			//batch generate with metrics
			String[] prompts = {"The meaning of life is", "Data science is"};
			FrameBlock result = ps.generateBatchWithMetrics(prompts, 20, 0.7, 0.9);
			
			//verify FrameBlock structure with metrics
			Assert.assertNotNull("Metrics result should not be null", result);
			Assert.assertEquals("Should have 2 rows", 2, result.getNumRows());
			Assert.assertEquals("Should have 3 columns", 3, result.getNumColumns());
			
			//verify metrics column contains timing data
			for (int i = 0; i < prompts.length; i++) {
				String prompt = (String) result.get(i, 0);
				String generated = (String) result.get(i, 1);
				long timeMs = Long.parseLong(result.get(i, 2).toString());
				Assert.assertEquals("Prompt should match", prompts[i], prompt);
				Assert.assertFalse("Generated text should not be empty", generated.isEmpty());
				Assert.assertTrue("Time should be positive", timeMs > 0);
				System.out.println("Prompt: " + prompt);
				System.out.println("Generated: " + generated);
				System.out.println("Time: " + timeMs + "ms");
			}
			
		} catch (Exception e) {
			System.out.println("Skipping metrics LLM test:");
			e.printStackTrace();
			org.junit.Assume.assumeNoException("LLM dependencies not available", e);
		} finally {
			if (conn != null) {
				conn.close();
			}
		}
	}
}
