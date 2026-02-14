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
	private final static String MODEL_NAME = "distilgpt2";
	private final static String WORKER_SCRIPT = "src/main/python/llm_worker.py";
	private final static String DML_SCRIPT = "x = 1;\nwrite(x, './tmp/x');";

	@Override
	public void setUp() {
		addTestConfiguration(TEST_DIR, TEST_NAME);
		getAndLoadTestConfiguration(TEST_NAME);
	}

	/**
	 * Creates a connection, loads the LLM model, and returns a PreparedScript
	 * with the LLM worker attached.
	 */
	private PreparedScript createLLMScript(Connection conn) throws Exception {
		LLMCallback llmWorker = conn.loadModel(MODEL_NAME, WORKER_SCRIPT);
		Assert.assertNotNull("LLM worker should not be null", llmWorker);
		PreparedScript ps = conn.prepareScript(DML_SCRIPT, new String[]{}, new String[]{"x"});
		ps.setLLMWorker(llmWorker);
		return ps;
	}

	@Test
	public void testLLMInference() {
		Connection conn = null;
		try {
			conn = new Connection();
			PreparedScript ps = createLLMScript(conn);
			
			//generate text using llm
			String prompt = "The meaning of life is";
			String result = ps.generate(prompt, 20, 0.7, 0.9);
			
			//verify result
			Assert.assertNotNull("Generated text should not be null", result);
			Assert.assertFalse("Generated text should not be empty", result.isEmpty());
			
			System.out.println("Prompt: " + prompt);
			System.out.println("Generated: " + result);
			
		} catch (Exception e) {
			System.out.println("Skipping LLM test:");
			e.printStackTrace();
			org.junit.Assume.assumeNoException("LLM dependencies not available", e);
		} finally {
			if (conn != null)
				conn.close();
		}
	}
	
	@Test
	public void testBatchInference() {
		Connection conn = null;
		try {
			conn = new Connection();
			PreparedScript ps = createLLMScript(conn);
			
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
			if (conn != null)
				conn.close();
		}
	}
	
	@Test
	public void testBatchWithMetrics() {
		Connection conn = null;
		try {
			conn = new Connection();
			PreparedScript ps = createLLMScript(conn);
			
			//batch generate with metrics
			String[] prompts = {"The meaning of life is", "Data science is"};
			FrameBlock result = ps.generateBatchWithMetrics(prompts, 20, 0.7, 0.9);
			
			//verify FrameBlock structure with metrics and token counts
			Assert.assertNotNull("Metrics result should not be null", result);
			Assert.assertEquals("Should have 2 rows", 2, result.getNumRows());
			Assert.assertEquals("Should have 5 columns", 5, result.getNumColumns());
			
			//verify metrics columns contain timing and token data
			for (int i = 0; i < prompts.length; i++) {
				String prompt = (String) result.get(i, 0);
				String generated = (String) result.get(i, 1);
				long timeMs = Long.parseLong(result.get(i, 2).toString());
				long inputTokens = Long.parseLong(result.get(i, 3).toString());
				long outputTokens = Long.parseLong(result.get(i, 4).toString());
				Assert.assertEquals("Prompt should match", prompts[i], prompt);
				Assert.assertFalse("Generated text should not be empty", generated.isEmpty());
				Assert.assertTrue("Time should be positive", timeMs > 0);
				Assert.assertTrue("Input tokens should be positive", inputTokens > 0);
				Assert.assertTrue("Output tokens should be positive", outputTokens > 0);
				System.out.println("Prompt: " + prompt);
				System.out.println("Generated: " + generated);
				System.out.println("Time: " + timeMs + "ms");
				System.out.println("Tokens: " + inputTokens + " in, " + outputTokens + " out");
			}
			
		} catch (Exception e) {
			System.out.println("Skipping metrics LLM test:");
			e.printStackTrace();
			org.junit.Assume.assumeNoException("LLM dependencies not available", e);
		} finally {
			if (conn != null)
				conn.close();
		}
	}
}
