package org.apache.sysds.api.jmlc;

/**
 * Interface for the Python LLM worker.
 * The Python side implements this via Py4J callback.
 */
public interface LLMCallback {
	
	/**
	 * Generates text using the LLM model.
	 * 
	 * @param prompt the input prompt text
	 * @param maxNewTokens maximum number of new tokens to generate
	 * @param temperature sampling temperature (0.0 = deterministic, higher = more random)
	 * @param topP nucleus sampling probability threshold
	 * @return generated text continuation
	 */
	String generate(String prompt, int maxNewTokens, double temperature, double topP);
	
	/**
	 * Generates text and returns result with token counts as a JSON string.
	 * Format: {"text": "...", "input_tokens": N, "output_tokens": M}
	 * 
	 * @param prompt the input prompt text
	 * @param maxNewTokens maximum number of new tokens to generate
	 * @param temperature sampling temperature (0.0 = deterministic, higher = more random)
	 * @param topP nucleus sampling probability threshold
	 * @return JSON string with generated text and token counts
	 */
	String generateWithTokenCount(String prompt, int maxNewTokens, double temperature, double topP);
}