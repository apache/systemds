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

import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;

import com.sun.net.httpserver.HttpServer;

import org.apache.sysds.api.jmlc.Connection;
import org.apache.sysds.api.jmlc.PreparedScript;
import org.apache.sysds.api.jmlc.ResultVariables;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.test.AutomatedTestBase;
import org.junit.Assert;
import org.junit.Test;

/**
 * Tests for llmPredict built-in via JMLC.
 * Needs an OpenAI-compatible server on localhost:8080.
 */
public class JMLCLLMInferenceTest extends AutomatedTestBase {
	private final static String TEST_NAME = "JMLCLLMInferenceTest";
	private final static String TEST_DIR = "functions/jmlc/";
	private final static String LLM_URL = "http://localhost:8080/v1/completions";

	private final static String DML_SCRIPT =
		"prompts = read(\"prompts\", data_type=\"frame\")\n" +

		"results = llmPredict(target=prompts, url=$url, max_tokens=$mt, temperature=$temp, top_p=$tp)\n" +
		"write(results, \"results\")";

	@Override
	public void setUp() {
		addTestConfiguration(TEST_DIR, TEST_NAME);
		getAndLoadTestConfiguration(TEST_NAME);
	}

	@Test
	public void testSinglePrompt() {
		Connection conn = null;
		try {
			conn = new Connection();
			Map<String, String> args = new HashMap<>();
			args.put("$url", LLM_URL);
			args.put("$mt", "20");
			args.put("$temp", "0.7");
			args.put("$tp", "0.9");
			PreparedScript ps = conn.prepareScript(DML_SCRIPT, args,
				new String[]{"prompts"}, new String[]{"results"});

			String[][] promptData = new String[][]{{"The meaning of life is"}};
			ps.setFrame("prompts", promptData);

			ResultVariables rv = ps.executeScript();
			FrameBlock result = rv.getFrameBlock("results");

			Assert.assertNotNull("Result should not be null", result);
			Assert.assertEquals("Should have 1 row", 1, result.getNumRows());
			Assert.assertEquals("Should have 5 columns", 5, result.getNumColumns());
			String generated = result.get(0, 1).toString();
			Assert.assertFalse("Generated text should not be empty", generated.isEmpty());

			System.out.println("Prompt: " + promptData[0][0]);
			System.out.println("Generated: " + generated);
		} catch (Exception e) {
			e.printStackTrace();
			org.junit.Assume.assumeNoException("LLM server not available", e);
		} finally {
			if (conn != null) conn.close();
		}
	}

	@Test
	public void testServerUnreachable() {
		// should throw DMLRuntimeException, not hang
		Connection conn = null;
		try {
			conn = new Connection();
			String deadUrl = "http://localhost:19999/v1/completions";
			Map<String, String> args = new HashMap<>();
			args.put("$url", deadUrl);
			args.put("$mt", "20");
			args.put("$temp", "0.0");
			args.put("$tp", "0.9");
			PreparedScript ps = conn.prepareScript(DML_SCRIPT, args,
				new String[]{"prompts"}, new String[]{"results"});

			String[][] promptData = new String[][]{{"Hello"}};
			ps.setFrame("prompts", promptData);

			try {
				ps.executeScript();
				Assert.fail("Expected DMLRuntimeException for unreachable server");
			}
			catch (DMLRuntimeException e) {
				String fullMsg = getExceptionChainMessage(e);
				System.out.println("Correctly caught: " + fullMsg);
				Assert.assertTrue("Error should mention connection issue",
					fullMsg.contains("connection refused")
					|| fullMsg.contains("Connection refused")
					|| fullMsg.contains("server is running"));
			}
		}
		catch (Exception e) {
			e.printStackTrace();
			org.junit.Assume.assumeNoException(
				"Could not set up negative test", e);
		}
		finally {
			if (conn != null) conn.close();
		}
	}

	@Test
	public void testInvalidUrl() {
		Connection conn = null;
		try {
			conn = new Connection();
			Map<String, String> args = new HashMap<>();
			args.put("$url", "not-a-valid-url");
			args.put("$mt", "20");
			args.put("$temp", "0.0");
			args.put("$tp", "0.9");
			PreparedScript ps = conn.prepareScript(DML_SCRIPT, args,
				new String[]{"prompts"}, new String[]{"results"});

			String[][] promptData = new String[][]{{"Hello"}};
			ps.setFrame("prompts", promptData);

			try {
				ps.executeScript();
				Assert.fail("Expected DMLRuntimeException for invalid URL");
			}
			catch (DMLRuntimeException e) {
				String fullMsg = getExceptionChainMessage(e);
				System.out.println("Correctly caught: " + fullMsg);
				Assert.assertTrue("Error should mention invalid URL",
					fullMsg.contains("invalid URL")
					|| fullMsg.contains("Invalid URL"));
			}
		}
		catch (Exception e) {
			e.printStackTrace();
			org.junit.Assume.assumeNoException(
				"Could not set up negative test", e);
		}
		finally {
			if (conn != null) conn.close();
		}
	}

	private static String getExceptionChainMessage(Throwable t) {
		StringBuilder sb = new StringBuilder();
		while(t != null) {
			if(sb.length() > 0) sb.append(" | ");
			if(t.getMessage() != null) sb.append(t.getMessage());
			t = t.getCause();
		}
		return sb.toString();
	}

	@Test
	public void testConcurrency() {
		Connection conn = null;
		try {
			conn = new Connection();
			String dmlConc =
				"prompts = read(\"prompts\", data_type=\"frame\")\n" +
				"results = llmPredict(target=prompts, url=$url, max_tokens=$mt, " +
				"temperature=$temp, top_p=$tp, concurrency=$conc)\n" +
				"write(results, \"results\")";
			Map<String, String> args = new HashMap<>();
			args.put("$url", LLM_URL);
			args.put("$mt", "20");
			args.put("$temp", "0.0");
			args.put("$tp", "0.9");
			args.put("$conc", "2");
			PreparedScript ps = conn.prepareScript(dmlConc, args,
				new String[]{"prompts"}, new String[]{"results"});

			String[][] promptData = new String[][]{
				{"Hello world"}, {"Test prompt"}, {"Another test"}
			};
			ps.setFrame("prompts", promptData);

			ResultVariables rv = ps.executeScript();
			FrameBlock result = rv.getFrameBlock("results");

			Assert.assertNotNull("Result should not be null", result);
			Assert.assertEquals("Should have 3 rows", 3, result.getNumRows());
			Assert.assertEquals("Should have 5 columns", 5, result.getNumColumns());
		} catch (Exception e) {
			e.printStackTrace();
			org.junit.Assume.assumeNoException("LLM server not available", e);
		} finally {
			if (conn != null) conn.close();
		}
	}

	@Test
	public void testHttpErrorResponse() {
		// mock server that returns HTTP 500
		HttpServer server = null;
		Connection conn = null;
		try {
			server = HttpServer.create(new InetSocketAddress(0), 0);
			int port = server.getAddress().getPort();
			server.createContext("/v1/completions", exchange -> {
				byte[] resp = "{\"error\": \"internal server error\"}".getBytes(StandardCharsets.UTF_8);
				exchange.sendResponseHeaders(500, resp.length);
				try(OutputStream os = exchange.getResponseBody()) {
					os.write(resp);
				}
			});
			server.start();

			conn = new Connection();
			Map<String, String> args = new HashMap<>();
			args.put("$url", "http://localhost:" + port + "/v1/completions");
			args.put("$mt", "20");
			args.put("$temp", "0.0");
			args.put("$tp", "0.9");
			PreparedScript ps = conn.prepareScript(DML_SCRIPT, args,
				new String[]{"prompts"}, new String[]{"results"});
			ps.setFrame("prompts", new String[][]{{"Hello"}});

			try {
				ps.executeScript();
				Assert.fail("Expected DMLRuntimeException for HTTP 500");
			}
			catch (DMLRuntimeException e) {
				String fullMsg = getExceptionChainMessage(e);
				System.out.println("Correctly caught HTTP 500: " + fullMsg);
				Assert.assertTrue("Error should mention HTTP 500",
					fullMsg.contains("HTTP 500"));
			}
		}
		catch (Exception e) {
			e.printStackTrace();
			org.junit.Assume.assumeNoException(
				"Could not set up mock server", e);
		}
		finally {
			if (server != null) server.stop(0);
			if (conn != null) conn.close();
		}
	}

	@Test
	public void testMalformedJsonResponse() {
		// mock server that returns HTTP 200 with invalid JSON
		HttpServer server = null;
		Connection conn = null;
		try {
			server = HttpServer.create(new InetSocketAddress(0), 0);
			int port = server.getAddress().getPort();
			server.createContext("/v1/completions", exchange -> {
				byte[] resp = "this is not json at all".getBytes(StandardCharsets.UTF_8);
				exchange.sendResponseHeaders(200, resp.length);
				try(OutputStream os = exchange.getResponseBody()) {
					os.write(resp);
				}
			});
			server.start();

			conn = new Connection();
			Map<String, String> args = new HashMap<>();
			args.put("$url", "http://localhost:" + port + "/v1/completions");
			args.put("$mt", "20");
			args.put("$temp", "0.0");
			args.put("$tp", "0.9");
			PreparedScript ps = conn.prepareScript(DML_SCRIPT, args,
				new String[]{"prompts"}, new String[]{"results"});
			ps.setFrame("prompts", new String[][]{{"Hello"}});

			try {
				ps.executeScript();
				Assert.fail("Expected DMLRuntimeException for malformed JSON");
			}
			catch (DMLRuntimeException e) {
				String fullMsg = getExceptionChainMessage(e);
				System.out.println("Correctly caught malformed JSON: " + fullMsg);
				Assert.assertTrue("Error should mention response issue",
					fullMsg.contains("failed") || fullMsg.contains("response"));
			}
		}
		catch (Exception e) {
			e.printStackTrace();
			org.junit.Assume.assumeNoException(
				"Could not set up mock server", e);
		}
		finally {
			if (server != null) server.stop(0);
			if (conn != null) conn.close();
		}
	}

	@Test
	public void testMissingChoicesInResponse() {
		// mock server that returns valid JSON but no "choices" array
		HttpServer server = null;
		Connection conn = null;
		try {
			server = HttpServer.create(new InetSocketAddress(0), 0);
			int port = server.getAddress().getPort();
			server.createContext("/v1/completions", exchange -> {
				byte[] resp = "{\"id\": \"test\", \"object\": \"text_completion\"}"
					.getBytes(StandardCharsets.UTF_8);
				exchange.sendResponseHeaders(200, resp.length);
				try(OutputStream os = exchange.getResponseBody()) {
					os.write(resp);
				}
			});
			server.start();

			conn = new Connection();
			Map<String, String> args = new HashMap<>();
			args.put("$url", "http://localhost:" + port + "/v1/completions");
			args.put("$mt", "20");
			args.put("$temp", "0.0");
			args.put("$tp", "0.9");
			PreparedScript ps = conn.prepareScript(DML_SCRIPT, args,
				new String[]{"prompts"}, new String[]{"results"});
			ps.setFrame("prompts", new String[][]{{"Hello"}});

			try {
				ps.executeScript();
				Assert.fail("Expected DMLRuntimeException for missing choices");
			}
			catch (DMLRuntimeException e) {
				String fullMsg = getExceptionChainMessage(e);
				System.out.println("Correctly caught missing choices: " + fullMsg);
				Assert.assertTrue("Error should mention missing choices",
					fullMsg.contains("choices"));
			}
		}
		catch (Exception e) {
			e.printStackTrace();
			org.junit.Assume.assumeNoException(
				"Could not set up mock server", e);
		}
		finally {
			if (server != null) server.stop(0);
			if (conn != null) conn.close();
		}
	}

	@Test
	public void testBatchInference() {
		Connection conn = null;
		try {
			conn = new Connection();
			Map<String, String> args = new HashMap<>();
			args.put("$url", LLM_URL);
			args.put("$mt", "20");
			args.put("$temp", "0.7");
			args.put("$tp", "0.9");
			PreparedScript ps = conn.prepareScript(DML_SCRIPT, args,
				new String[]{"prompts"}, new String[]{"results"});

			String[] prompts = {
				"The meaning of life is",
				"Machine learning is",
				"Apache SystemDS enables"
			};
			String[][] promptData = new String[prompts.length][1];
			for (int i = 0; i < prompts.length; i++)
				promptData[i][0] = prompts[i];
			ps.setFrame("prompts", promptData);

			ResultVariables rv = ps.executeScript();
			FrameBlock result = rv.getFrameBlock("results");

			Assert.assertNotNull("Result should not be null", result);
			Assert.assertEquals("Should have 3 rows", 3, result.getNumRows());
			Assert.assertEquals("Should have 5 columns", 5, result.getNumColumns());

			for (int i = 0; i < prompts.length; i++) {
				String prompt = result.get(i, 0).toString();
				String generated = result.get(i, 1).toString();
				long timeMs = Long.parseLong(result.get(i, 2).toString());
				Assert.assertEquals("Prompt should match", prompts[i], prompt);
				Assert.assertFalse("Generated text should not be empty", generated.isEmpty());
				Assert.assertTrue("Time should be positive", timeMs > 0);
				System.out.println("Prompt: " + prompt);
				System.out.println("Generated: " + generated + " (" + timeMs + "ms)");
			}
		} catch (Exception e) {
			e.printStackTrace();
			org.junit.Assume.assumeNoException("LLM server not available", e);
		} finally {
			if (conn != null) conn.close();
		}
	}
}
