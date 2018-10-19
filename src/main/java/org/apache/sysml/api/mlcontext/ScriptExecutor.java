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

package org.apache.sysml.api.mlcontext;

import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.commons.lang3.StringUtils;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.api.ScriptExecutorUtils;
import org.apache.sysml.api.ScriptExecutorUtils.SystemMLAPI;
import org.apache.sysml.api.mlcontext.MLContext.ExecutionType;
import org.apache.sysml.api.mlcontext.MLContext.ExplainLevel;
import org.apache.sysml.conf.CompilerConfig;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.conf.DMLConfig;
import org.apache.sysml.conf.DMLOptions;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.parser.DMLProgram;
import org.apache.sysml.parser.DMLTranslator;
import org.apache.sysml.parser.ParseException;
import org.apache.sysml.parser.ParserFactory;
import org.apache.sysml.parser.ParserWrapper;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.LocalVariableMap;
import org.apache.sysml.runtime.controlprogram.Program;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.instructions.gpu.context.GPUContext;
import org.apache.sysml.runtime.instructions.gpu.context.GPUContextPool;
import org.apache.sysml.utils.Explain;
import org.apache.sysml.utils.Explain.ExplainType;
import org.apache.sysml.utils.Statistics;

/**
 * ScriptExecutor executes a DML or PYDML Script object using SystemML. This is
 * accomplished by calling the {@link #execute} method.
 * <p>
 * Script execution via the MLContext API typically consists of the following
 * steps:
 * </p>
 * <ol>
 * <li>Language Steps
 * <ol>
 * <li>Parse script into program</li>
 * <li>Live variable analysis</li>
 * <li>Validate program</li>
 * </ol>
 * </li>
 * <li>HOP (High-Level Operator) Steps
 * <ol>
 * <li>Construct HOP DAGs</li>
 * <li>Static rewrites</li>
 * <li>Intra-/Inter-procedural analysis</li>
 * <li>Dynamic rewrites</li>
 * <li>Compute memory estimates</li>
 * <li>Rewrite persistent reads and writes (MLContext-specific)</li>
 * </ol>
 * </li>
 * <li>LOP (Low-Level Operator) Steps
 * <ol>
 * <li>Contruct LOP DAGs</li>
 * <li>Generate runtime program</li>
 * <li>Execute runtime program</li>
 * <li>Dynamic recompilation</li>
 * </ol>
 * </li>
 * </ol>
 * <p>
 * Modifications to these steps can be accomplished by subclassing
 * ScriptExecutor. For more information, please see the {@link #execute} method.
 */
public class ScriptExecutor {

	protected DMLConfig config;
	protected DMLProgram dmlProgram;
	protected Program runtimeProgram;
	protected ExecutionContext executionContext;
	protected Script script;
	protected boolean init = false;
	protected boolean explain = false;
	protected boolean gpu = false;
	protected boolean oldGPU = false;
	protected boolean forceGPU = false;
	protected boolean oldForceGPU = false;
	protected boolean statistics = false;
	protected ExplainLevel explainLevel;
	protected ExecutionType executionType;
	protected int statisticsMaxHeavyHitters = 10;
	protected boolean maintainSymbolTable = false;
	protected List<GPUContext> gCtxs = null;

	/**
	 * ScriptExecutor constructor.
	 */
	public ScriptExecutor() {
		config = ConfigurationManager.getDMLConfig();
	}

	/**
	 * ScriptExecutor constructor, where the configuration properties are passed
	 * in.
	 *
	 * @param config
	 *            the configuration properties to use by the ScriptExecutor
	 */
	public ScriptExecutor(DMLConfig config) {
		this.config = config;
		ConfigurationManager.setGlobalConfig(config);
	}

	/**
	 * Set the global flags (for example: statistics, gpu, etc).
	 */
	protected void setGlobalFlags() {
		ConfigurationManager.setStatistics(statistics);
		oldForceGPU = ConfigurationManager.isForcedGPU();
		ConfigurationManager.getDMLOptions().setForceGPU(forceGPU);
		oldGPU = ConfigurationManager.isGPU();
		ConfigurationManager.getDMLOptions().setGPU(gpu);
		ConfigurationManager.getDMLOptions().setStatisticsMaxHeavyHitters(statisticsMaxHeavyHitters);

		// set the global compiler configuration
		try {
			OptimizerUtils.resetStaticCompilerFlags();
			CompilerConfig cconf = OptimizerUtils.constructCompilerConfig(
					ConfigurationManager.getCompilerConfig(), config);
			ConfigurationManager.setGlobalConfig(cconf);
		} 
		catch(DMLRuntimeException ex) {
			throw new RuntimeException(ex);
		}

		DMLScript.setGlobalFlags(config);
	}
	

	/**
	 * Reset the global flags (for example: statistics, gpu, etc)
	 * post-execution.
	 */
	protected void resetGlobalFlags() {
		ConfigurationManager.getDMLOptions().setForceGPU(oldForceGPU);
		ConfigurationManager.getDMLOptions().setGPU(oldGPU);
		ConfigurationManager.getDMLOptions().setStatisticsMaxHeavyHitters(DMLOptions.defaultOptions.statsCount);
	}
	
	public void compile(Script script) {
		compile(script, true);
	}
	
	/**
	 * Compile a DML or PYDML script. This will help analysis of DML programs
	 * that have dynamic recompilation flag set to false without actually executing it. 
	 *
	 * @param script
	 *            the DML or PYDML script to compile
	 * @param performHOPRewrites
	 *            should perform static rewrites, perform intra-/inter-procedural analysis to propagate size information into functions and apply dynamic rewrites
	 */
	public void compile(Script script, boolean performHOPRewrites) {

		setup(script);

		LocalVariableMap symbolTable = script.getSymbolTable();
		String[] inputs = null; String[] outputs = null;
		if (symbolTable != null) {
			inputs = (script.getInputVariables() == null) ? new String[0]
					: script.getInputVariables().toArray(new String[0]);
			outputs = (script.getOutputVariables() == null) ? new String[0]
					: script.getOutputVariables().toArray(new String[0]);
		}

		Map<String, String> args = MLContextUtil
				.convertInputParametersForParser(script.getInputParameters(), script.getScriptType());
		runtimeProgram = ScriptExecutorUtils.compileRuntimeProgram(script.getScriptExecutionString(), Collections.emptyMap(),
				args, null, symbolTable, inputs, outputs, script.getScriptType(), config, SystemMLAPI.MLContext,
				performHOPRewrites, isMaintainSymbolTable(), init);
		gCtxs = ConfigurationManager.isGPU() ? GPUContextPool.getAllGPUContexts() : null;
	}


	/**
	 * Execute a DML or PYDML script. This is broken down into the following
	 * primary methods:
	 *
	 * <ol>
	 * <li>{@link #compile(Script)}</li>
	 * <li>{@link #cleanupAfterExecution()}</li>
	 * </ol>
	 *
	 * @param script
	 *            the DML or PYDML script to execute
	 * @return the results as a MLResults object
	 */
	public MLResults execute(Script script) {

		Map<String, String> args = MLContextUtil
				.convertInputParametersForParser(script.getInputParameters(), script.getScriptType());
		
		Explain.ExplainType explainType = Explain.ExplainType.NONE;
		if(explain) {
			explainType = (explainLevel == null) ? Explain.ExplainType.RUNTIME : explainLevel.getExplainType();
		}
		RUNTIME_PLATFORM rtplatform = DMLOptions.defaultOptions.execMode;
		if(executionType != null) {
			rtplatform = getExecutionType().getRuntimePlatform();
		}
		ConfigurationManager.setGlobalOptions(new DMLOptions(args, 
				statistics, statisticsMaxHeavyHitters, false, explainType, 
				rtplatform, gpu, forceGPU, script.getScriptType(), DMLScript.DML_FILE_PATH_ANTLR_PARSER, 
				script.getScriptExecutionString()));

		// main steps in script execution
		compile(script);

		try {
			executionContext = ScriptExecutorUtils.executeRuntimeProgram(getRuntimeProgram(),
					statistics ? statisticsMaxHeavyHitters : 0, script.getSymbolTable(),
					new HashSet<>(getScript().getOutputVariables()), SystemMLAPI.MLContext, gCtxs);
		} catch (DMLRuntimeException e) {
			throw new MLContextException("Exception occurred while executing runtime program", e);
		} finally {
			cleanupAfterExecution();
		}

		// add symbol table to MLResults
		MLResults mlResults = new MLResults(script);
		script.setResults(mlResults);

		return mlResults;
	}

	/**
	 * Sets the script in the ScriptExecutor, checks that the script has a type
	 * and string, sets the ScriptExecutor in the script, sets the script string
	 * in the Spark Monitor, globally sets the script type, sets global flags,
	 * and resets statistics if needed.
	 *
	 * @param script
	 *            the DML or PYDML script to execute
	 */
	protected void setup(Script script) {
		this.script = script;
		if (script == null) {
			throw new MLContextException("Script is null");
		} else if (script.getScriptType() == null) {
			throw new MLContextException("ScriptType (DML or PYDML) needs to be specified");
		} else if (script.getScriptString() == null) {
			throw new MLContextException("Script string is null");
		} else if (StringUtils.isBlank(script.getScriptString())) {
			throw new MLContextException("Script string is blank");
		}
		script.setScriptExecutor(this);

		// Set global variable indicating the script type
		DMLScript.SCRIPT_TYPE = script.getScriptType();
		setGlobalFlags();
		// reset all relevant summary statistics
		Statistics.resetNoOfExecutedJobs();
		if (statistics)
			Statistics.reset();
		DMLScript.EXPLAIN = (explainLevel != null) ? explainLevel.getExplainType() : ExplainType.NONE;
	}

	/**
	 * Perform any necessary cleanup operations after program execution.
	 */
	protected void cleanupAfterExecution() {
		restoreInputsInSymbolTable();
		resetGlobalFlags();
	}
	
	/**
	 * Restore the input variables in the symbol table after script execution.
	 */
	protected void restoreInputsInSymbolTable() {
		Map<String, Object> inputs = script.getInputs();
		Map<String, Metadata> inputMetadata = script.getInputMetadata();
		LocalVariableMap symbolTable = script.getSymbolTable();
		Set<String> inputVariables = script.getInputVariables();
		for (String inputVariable : inputVariables) {
			if (symbolTable.get(inputVariable) == null) {
				// retrieve optional metadata if it exists
				Metadata m = inputMetadata.get(inputVariable);
				script.in(inputVariable, inputs.get(inputVariable), m);
			}
		}
	}

	/**
	 * Parse the script into an ANTLR parse tree, and convert this parse tree
	 * into a SystemML program. Parsing includes lexical/syntactic analysis.
	 */
	protected void parseScript() {
		try {
			ParserWrapper parser = ParserFactory.createParser(script.getScriptType());
			Map<String, String> args = MLContextUtil
					.convertInputParametersForParser(script.getInputParameters(), script.getScriptType());
			dmlProgram = parser.parse(null, script.getScriptExecutionString(), args);
		} catch (ParseException e) {
			throw new MLContextException("Exception occurred while parsing script", e);
		}
	}

	/**
	 * Set the SystemML configuration properties.
	 *
	 * @param config
	 *            The configuration properties
	 */
	public void setConfig(DMLConfig config) {
		this.config = config;
		ConfigurationManager.setGlobalConfig(config);
	}

	/**
	 * Obtain the program
	 *
	 * @return the program
	 */
	public DMLProgram getDmlProgram() {
		return dmlProgram;
	}

	/**
	 * Obtain the runtime program
	 *
	 * @return the runtime program
	 */
	public Program getRuntimeProgram() {
		return runtimeProgram;
	}

	/**
	 * Obtain the execution context
	 *
	 * @return the execution context
	 */
	public ExecutionContext getExecutionContext() {
		return executionContext;
	}

	/**
	 * Obtain the Script object associated with this ScriptExecutor
	 *
	 * @return the Script object associated with this ScriptExecutor
	 */
	public Script getScript() {
		return script;
	}

	/**
	 * Whether or not an explanation of the DML/PYDML program should be output
	 * to standard output.
	 *
	 * @param explain
	 *            {@code true} if explanation should be output, {@code false}
	 *            otherwise
	 */
	public void setExplain(boolean explain) { this.explain = explain; }

	/**
	 * Whether or not statistics about the DML/PYDML program should be output to
	 * standard output.
	 *
	 * @param statistics
	 *            {@code true} if statistics should be output, {@code false}
	 *            otherwise
	 */
	public void setStatistics(boolean statistics) {
		this.statistics = statistics;
	}

	/**
	 * Set the maximum number of heavy hitters to display with statistics.
	 *
	 * @param maxHeavyHitters
	 *            the maximum number of heavy hitters
	 */
	public void setStatisticsMaxHeavyHitters(int maxHeavyHitters) {
		this.statisticsMaxHeavyHitters = maxHeavyHitters;
	}

	/**
	 * Obtain whether or not all values should be maintained in the symbol table
	 * after execution.
	 *
	 * @return {@code true} if all values should be maintained in the symbol
	 *         table, {@code false} otherwise
	 */
	public boolean isMaintainSymbolTable() {
		return maintainSymbolTable;
	}

	/**
	 * Set whether or not all values should be maintained in the symbol table
	 * after execution.
	 *
	 * @param maintainSymbolTable
	 *            {@code true} if all values should be maintained in the symbol
	 *            table, {@code false} otherwise
	 */
	public void setMaintainSymbolTable(boolean maintainSymbolTable) {
		this.maintainSymbolTable = maintainSymbolTable;
	}

	/**
	 * Whether or not to initialize the scratch_space, bufferpool, etc. Note
	 * that any redundant initialize (e.g., multiple scripts from one MLContext)
	 * clears existing files from the scratch space and buffer pool.
	 *
	 * @param init
	 *            {@code true} if should initialize, {@code false} otherwise
	 */
	public void setInit(boolean init) {
		this.init = init;
	}

	/**
	 * Set the level of program explanation that should be displayed if explain
	 * is set to true.
	 *
	 * @param explainLevel
	 *            the level of program explanation
	 */
	public void setExplainLevel(ExplainLevel explainLevel) {
		this.explainLevel = explainLevel;
		if (explainLevel == null) {
			DMLScript.EXPLAIN = ExplainType.NONE;
		} else {
			DMLScript.EXPLAIN = explainLevel.getExplainType();
		}
	}

	/**
	 * Whether or not to enable GPU usage.
	 *
	 * @param enabled
	 *            {@code true} if enabled, {@code false} otherwise
	 */
	public void setGPU(boolean enabled) {
		this.gpu = enabled;
	}

	/**
	 * Whether or not to force GPU usage.
	 *
	 * @param enabled
	 *            {@code true} if enabled, {@code false} otherwise
	 */
	public void setForceGPU(boolean enabled) {
		this.forceGPU = enabled;
	}

	/**
	 * Obtain the SystemML configuration properties.
	 *
	 * @return the configuration properties
	 */
	public DMLConfig getConfig() {
		return config;
	}

	/**
	 * Obtain the current execution environment.
	 * 
	 * @return the execution environment
	 */
	public ExecutionType getExecutionType() {
		return executionType;
	}

	/**
	 * Set the execution environment.
	 * 
	 * @param executionType
	 *            the execution environment
	 */
	public void setExecutionType(ExecutionType executionType) {
		ConfigurationManager.getDMLOptions().setExecutionMode(executionType.getRuntimePlatform());
		this.executionType = executionType;
	}
}
