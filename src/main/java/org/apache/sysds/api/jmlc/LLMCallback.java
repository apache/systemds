package org.apache.sysds.api.jmlc;

/**
 * Interface for the Python LLM worker.
 * The Python side implements this via Py4J callback.
 */
public interface LLMCallback {
	
	/**
	 * Generates text using the LLM model.
	 */
	String generate(String prompt, int maxNewTokens, double temperature, double topP);
	
	/**
	 * Generates text and returns result with token counts as a JSON string.
	 * Format: {"text": "...", "input_tokens": N, "output_tokens": M}
	 */
	String generateWithTokenCount(String prompt, int maxNewTokens, double temperature, double topP);
	
	/**
	 * Generates text for multiple prompts in a single batched GPU call.
	 * Returns a JSON array of objects with text and token counts.
	 * Format: [{"text": "...", "input_tokens": N, "output_tokens": M, "time_ms": T}, ...]
	 *
	 * @param prompts array of input prompt texts
	 * @param maxNewTokens maximum number of new tokens to generate per prompt
	 * @param temperature sampling temperature
	 * @param topP nucleus sampling probability threshold
	 * @return JSON array string with results for each prompt
	 */
	String generateBatch(String[] prompts, int maxNewTokens, double temperature, double topP);
}
