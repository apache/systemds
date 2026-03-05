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

package org.apache.sysds.runtime.instructions.cp;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.ConnectException;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.SocketTimeoutException;
import java.net.URI;
import java.net.URISyntaxException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.lineage.LineageItemUtils;
import org.apache.wink.json4j.JSONObject;

public class LlmPredictCPInstruction extends ParameterizedBuiltinCPInstruction {

	protected LlmPredictCPInstruction(LinkedHashMap<String, String> paramsMap,
			CPOperand out, String opcode, String istr) {
		super(null, paramsMap, out, opcode, istr);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		FrameBlock prompts = ec.getFrameInput(params.get("target"));
		String url = params.get("url");
		String model = params.containsKey("model") ?
			params.get("model") : null;
		int maxTokens = params.containsKey("max_tokens") ?
			Integer.parseInt(params.get("max_tokens")) : 512;
		double temperature = params.containsKey("temperature") ?
			Double.parseDouble(params.get("temperature")) : 0.0;
		double topP = params.containsKey("top_p") ?
			Double.parseDouble(params.get("top_p")) : 0.9;
		int concurrency = params.containsKey("concurrency") ?
			Integer.parseInt(params.get("concurrency")) : 1;
		concurrency = Math.max(1, Math.min(concurrency, 128));

		int n = prompts.getNumRows();
		String[][] data = new String[n][];

		List<Callable<String[]>> tasks = new ArrayList<>(n);
		for(int i = 0; i < n; i++) {
			String prompt = prompts.get(i, 0).toString();
			tasks.add(() -> callLlmEndpoint(prompt, url, model, maxTokens, temperature, topP));
		}

		try {
			if(concurrency <= 1) {
				for(int i = 0; i < n; i++)
					data[i] = tasks.get(i).call();
			}
			else {
				ExecutorService pool = Executors.newFixedThreadPool(
					Math.min(concurrency, n));
				List<Future<String[]>> futures = pool.invokeAll(tasks);
				pool.shutdown();
				for(int i = 0; i < n; i++)
					data[i] = futures.get(i).get();
			}
		}
		catch(DMLRuntimeException e) {
			throw e;
		}
		catch(Exception e) {
			throw new DMLRuntimeException("llmPredict failed: " + e.getMessage(), e);
		}

		ValueType[] schema = {ValueType.STRING, ValueType.STRING,
			ValueType.INT64, ValueType.INT64, ValueType.INT64};
		String[] colNames = {"prompt", "generated_text", "time_ms", "input_tokens", "output_tokens"};
		FrameBlock fbout = new FrameBlock(schema, colNames);
		for(String[] row : data)
			fbout.appendRow(row);

		ec.setFrameOutput(output.getName(), fbout);
		ec.releaseFrameInput(params.get("target"));
	}

	private static String[] callLlmEndpoint(String prompt, String url,
			String model, int maxTokens, double temperature, double topP) {
		long t0 = System.nanoTime();

		// validate URL and open connection
		HttpURLConnection conn;
		try {
			conn = (HttpURLConnection) new URI(url).toURL().openConnection();
		}
		catch(URISyntaxException | MalformedURLException | IllegalArgumentException e) {
			throw new DMLRuntimeException(
				"llmPredict: invalid URL '" + url + "'. "
				+ "Expected format: http://host:port/v1/completions", e);
		}
		catch(IOException e) {
			throw new DMLRuntimeException(
				"llmPredict: cannot open connection to '" + url + "'.", e);
		}

		try {
			JSONObject req = new JSONObject();
			if(model != null)
				req.put("model", model);
			req.put("prompt", prompt);
			req.put("max_tokens", maxTokens);
			req.put("temperature", temperature);
			req.put("top_p", topP);

			conn.setRequestMethod("POST");
			conn.setRequestProperty("Content-Type", "application/json");
			conn.setConnectTimeout(10_000);
			conn.setReadTimeout(300_000);
			conn.setDoOutput(true);

			try(OutputStream os = conn.getOutputStream()) {
				os.write(req.toString().getBytes(StandardCharsets.UTF_8));
			}

			int httpCode = conn.getResponseCode();
			if(httpCode != 200) {
				String errBody = "";
				try(InputStream es = conn.getErrorStream()) {
					if(es != null)
						errBody = new String(es.readAllBytes(), StandardCharsets.UTF_8);
				}
				catch(Exception ignored) {}
				throw new DMLRuntimeException(
					"llmPredict: endpoint returned HTTP " + httpCode
					+ " for '" + url + "'."
					+ (errBody.isEmpty() ? "" : " Response: " + errBody));
			}

			String body;
			try(InputStream is = conn.getInputStream()) {
				body = new String(is.readAllBytes(), StandardCharsets.UTF_8);
			}

			JSONObject resp = new JSONObject(body);
			if(!resp.has("choices") || resp.getJSONArray("choices").length() == 0) {
				String errMsg = resp.has("error") ? resp.optString("error") : body;
				throw new DMLRuntimeException(
					"llmPredict: server response missing 'choices'. Response: " + errMsg);
			}
			String text = resp.getJSONArray("choices")
				.getJSONObject(0).getString("text");
			long elapsed = (System.nanoTime() - t0) / 1_000_000;
			int inTok = 0, outTok = 0;
			if(resp.has("usage")) {
				JSONObject usage = resp.getJSONObject("usage");
				inTok = usage.has("prompt_tokens") ? usage.getInt("prompt_tokens") : 0;
				outTok = usage.has("completion_tokens") ? usage.getInt("completion_tokens") : 0;
			}
			return new String[]{prompt, text,
				String.valueOf(elapsed), String.valueOf(inTok), String.valueOf(outTok)};
		}
		catch(ConnectException e) {
			throw new DMLRuntimeException(
				"llmPredict: connection refused to '" + url + "'. "
				+ "Ensure the LLM server is running and reachable.", e);
		}
		catch(SocketTimeoutException e) {
			throw new DMLRuntimeException(
				"llmPredict: timed out connecting to '" + url + "'. "
				+ "Ensure the LLM server is running and reachable.", e);
		}
		catch(IOException e) {
			throw new DMLRuntimeException(
				"llmPredict: I/O error communicating with '" + url + "'.", e);
		}
		catch(DMLRuntimeException e) {
			throw e;
		}
		catch(Exception e) {
			throw new DMLRuntimeException(
				"llmPredict: failed to get response from '" + url + "'.", e);
		}
		finally {
			conn.disconnect();
		}
	}

	@Override
	public Pair<String, LineageItem> getLineageItem(ExecutionContext ec) {
		CPOperand target = new CPOperand(params.get("target"), ValueType.STRING, DataType.FRAME);
		CPOperand urlOp = new CPOperand(params.get("url"), ValueType.STRING, DataType.SCALAR, true);
		return Pair.of(output.getName(),
			new LineageItem(getOpcode(), LineageItemUtils.getLineage(ec, target, urlOp)));
	}
}
