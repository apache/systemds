package org.apache.sysds.api.jmlc;

/**
 * Interface for the Python LLM worker.
 * The Python side implements this via Py4J callback.
 */
public interface LLMCallback {
    String generate(String prompt, int maxNewTokens, double temperature, double topP);
}