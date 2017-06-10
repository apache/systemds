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

import java.io.IOException;
import java.util.Map;
import java.util.Set;

import org.apache.commons.lang3.StringUtils;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.ScriptExecutorUtils;
import org.apache.sysml.api.DMLScript.DMLOptions;
import org.apache.sysml.api.jmlc.JMLCUtils;
import org.apache.sysml.api.mlcontext.MLContext.ExplainLevel;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.conf.DMLConfig;
import org.apache.sysml.hops.HopsException;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.hops.OptimizerUtils.OptimizationLevel;
import org.apache.sysml.hops.globalopt.GlobalOptimizerWrapper;
import org.apache.sysml.hops.rewrite.ProgramRewriter;
import org.apache.sysml.hops.rewrite.RewriteRemovePersistentReadWrite;
import org.apache.sysml.lops.LopsException;
import org.apache.sysml.parser.DMLProgram;
import org.apache.sysml.parser.DMLTranslator;
import org.apache.sysml.parser.LanguageException;
import org.apache.sysml.parser.ParseException;
import org.apache.sysml.parser.ParserFactory;
import org.apache.sysml.parser.ParserWrapper;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.LocalVariableMap;
import org.apache.sysml.runtime.controlprogram.Program;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContextFactory;
import org.apache.sysml.utils.Explain;
import org.apache.sysml.utils.Explain.ExplainCounts;
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
 * ScriptExecutor. For example, the following code will turn off the global data
 * flow optimization check by subclassing ScriptExecutor and overriding the
 * globalDataFlowOptimization method.
 * </p>
 *
 * <code>ScriptExecutor scriptExecutor = new ScriptExecutor() {
 * <br>&nbsp;&nbsp;// turn off global data flow optimization check
 * <br>&nbsp;&nbsp;@Override
 * <br>&nbsp;&nbsp;protected void globalDataFlowOptimization() {
 * <br>&nbsp;&nbsp;&nbsp;&nbsp;return;
 * <br>&nbsp;&nbsp;}
 * <br>};
 * <br>ml.execute(script, scriptExecutor);</code>
 * <p>
 *
 * For more information, please see the {@link #execute} method.
 */
public class ScriptExecutor {

	protected DMLConfig config;
	protected DMLProgram dmlProgram;
	protected DMLTranslator dmlTranslator;
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
	protected boolean oldStatistics = false;
	protected ExplainLevel explainLevel;
	protected int statisticsMaxHeavyHitters = 10;
	protected boolean maintainSymbolTable = false;

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
	 * Construct DAGs of high-level operators (HOPs) for each block of
	 * statements.
	 */
	protected void constructHops() {
		try {
			dmlTranslator.constructHops(dmlProgram);
		} catch (LanguageException | ParseException e) {
			throw new MLContextException("Exception occurred while constructing HOPS (high-level operators)", e);
		}
	}

	/**
	 * Apply static rewrites, perform intra-/inter-procedural analysis to
	 * propagate size information into functions, apply dynamic rewrites, and
	 * compute memory estimates for all HOPs.
	 */
	protected void rewriteHops() {
		try {
			dmlTranslator.rewriteHopsDAG(dmlProgram);
		} catch (LanguageException | HopsException | ParseException e) {
			throw new MLContextException("Exception occurred while rewriting HOPS (high-level operators)", e);
		}
	}

	/**
	 * Output a description of the program to standard output.
	 */
	protected void showExplanation() {
		if (!explain)
			return;

		try {
			ExplainType explainType = (explainLevel != null) ? explainLevel.getExplainType() : ExplainType.RUNTIME;
			System.out.println(Explain.explain(dmlProgram, runtimeProgram, explainType));
		} catch (Exception e) {
			throw new MLContextException("Exception occurred while explaining dml program", e);
		}
	}

	/**
	 * Construct DAGs of low-level operators (LOPs) based on the DAGs of
	 * high-level operators (HOPs).
	 */
	protected void constructLops() {
		try {
			dmlTranslator.constructLops(dmlProgram);
		} catch (ParseException | LanguageException | HopsException | LopsException e) {
			throw new MLContextException("Exception occurred while constructing LOPS (low-level operators)", e);
		}
	}

	/**
	 * Create runtime program. For each namespace, translate function statement
	 * blocks into function program blocks and add these to the runtime program.
	 * For each top-level block, add the program block to the runtime program.
	 */
	protected void generateRuntimeProgram() {
		try {
			runtimeProgram = dmlProgram.getRuntimeProgram(config);
		} catch (LanguageException | DMLRuntimeException | LopsException | IOException e) {
			throw new MLContextException("Exception occurred while generating runtime program", e);
		}
	}

	/**
	 * Count the number of compiled MR Jobs/Spark Instructions in the runtime
	 * program and set this value in the statistics.
	 */
	protected void countCompiledMRJobsAndSparkInstructions() {
		ExplainCounts counts = Explain.countDistributedOperations(runtimeProgram);
		Statistics.resetNoOfCompiledJobs(counts.numJobs);
	}

	/**
	 * Create an execution context and set its variables to be the symbol table
	 * of the script.
	 */
	protected void createAndInitializeExecutionContext() {
		executionContext = ExecutionContextFactory.createContext(runtimeProgram);
		LocalVariableMap symbolTable = script.getSymbolTable();
		if (symbolTable != null) {
			executionContext.setVariables(symbolTable);
		}

	}

	/**
	 * Set the global flags (for example: statistics, gpu, etc).
	 */
	protected void setGlobalFlags() {
		oldStatistics = DMLScript.STATISTICS;
		DMLScript.STATISTICS = statistics;
		oldForceGPU = DMLScript.FORCE_ACCELERATOR;
		DMLScript.FORCE_ACCELERATOR = forceGPU;
		oldGPU = DMLScript.USE_ACCELERATOR;
		DMLScript.USE_ACCELERATOR = gpu;
		DMLScript.STATISTICS_COUNT = statisticsMaxHeavyHitters;
	}

	/**
	 * Reset the global flags (for example: statistics, gpu, etc)
	 * post-execution.
	 */
	protected void resetGlobalFlags() {
		DMLScript.STATISTICS = oldStatistics;
		DMLScript.FORCE_ACCELERATOR = oldForceGPU;
		DMLScript.USE_ACCELERATOR = oldGPU;
		DMLScript.STATISTICS_COUNT = DMLOptions.defaultOptions.statsCount;
	}

	/**
	 * Execute a DML or PYDML script. This is broken down into the following
	 * primary methods:
	 *
	 * <ol>
	 * <li>{@link #setup(Script)}</li>
	 * <li>{@link #parseScript()}</li>
	 * <li>{@link #liveVariableAnalysis()}</li>
	 * <li>{@link #validateScript()}</li>
	 * <li>{@link #constructHops()}</li>
	 * <li>{@link #rewriteHops()}</li>
	 * <li>{@link #rewritePersistentReadsAndWrites()}</li>
	 * <li>{@link #constructLops()}</li>
	 * <li>{@link #generateRuntimeProgram()}</li>
	 * <li>{@link #showExplanation()}</li>
	 * <li>{@link #globalDataFlowOptimization()}</li>
	 * <li>{@link #countCompiledMRJobsAndSparkInstructions()}</li>
	 * <li>{@link #initializeCachingAndScratchSpace()}</li>
	 * <li>{@link #cleanupRuntimeProgram()}</li>
	 * <li>{@link #createAndInitializeExecutionContext()}</li>
	 * <li>{@link #executeRuntimeProgram()}</li>
	 * <li>{@link #cleanupAfterExecution()}</li>
	 * </ol>
	 *
	 * @param script
	 *            the DML or PYDML script to execute
	 * @return the results as a MLResults object
	 */
	public MLResults execute(Script script) {

		// main steps in script execution
		setup(script);
		parseScript();
		liveVariableAnalysis();
		validateScript();
		constructHops();
		rewriteHops();
		rewritePersistentReadsAndWrites();
		constructLops();
		generateRuntimeProgram();
		showExplanation();
		globalDataFlowOptimization();
		countCompiledMRJobsAndSparkInstructions();
		initializeCachingAndScratchSpace();
		cleanupRuntimeProgram();

		try {
			createAndInitializeExecutionContext();
			executeRuntimeProgram();
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
		checkScriptHasTypeAndString();
		script.setScriptExecutor(this);
		// Set global variable indicating the script type
		DMLScript.SCRIPT_TYPE = script.getScriptType();
		setGlobalFlags();
		if (statistics) {
			Statistics.reset();
		}
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
	 * If {@code maintainSymbolTable} is true, delete all 'remove variable'
	 * instructions so as to maintain the values in the symbol table, which are
	 * useful when working interactively in an environment such as the Spark
	 * Shell. Otherwise, only delete 'remove variable' instructions for
	 * registered outputs.
	 */
	protected void cleanupRuntimeProgram() {
		if (maintainSymbolTable) {
			MLContextUtil.deleteRemoveVariableInstructions(runtimeProgram);
		} else {
			JMLCUtils.cleanupRuntimeProgram(runtimeProgram, (script.getOutputVariables() == null) ? new String[0]
					: script.getOutputVariables().toArray(new String[0]));
		}
	}

	/**
	 * Execute the runtime program. This involves execution of the program
	 * blocks that make up the runtime program and may involve dynamic
	 * recompilation.
	 */
	protected void executeRuntimeProgram() {
		try {
			ScriptExecutorUtils.executeRuntimeProgram(this, statistics ? statisticsMaxHeavyHitters : 0);
		} catch (DMLRuntimeException e) {
			throw new MLContextException("Exception occurred while executing runtime program", e);
		}
	}

	/**
	 * Check security, create scratch space, cleanup working directories,
	 * initialize caching, and reset statistics.
	 */
	protected void initializeCachingAndScratchSpace() {
		if (!init)
			return;

		try {
			DMLScript.initHadoopExecution(config);
		} catch (ParseException e) {
			throw new MLContextException("Exception occurred initializing caching and scratch space", e);
		} catch (DMLRuntimeException e) {
			throw new MLContextException("Exception occurred initializing caching and scratch space", e);
		} catch (IOException e) {
			throw new MLContextException("Exception occurred initializing caching and scratch space", e);
		}
	}

	/**
	 * Optimize the program.
	 */
	protected void globalDataFlowOptimization() {
		if (OptimizerUtils.isOptLevel(OptimizationLevel.O4_GLOBAL_TIME_MEMORY)) {
			try {
				runtimeProgram = GlobalOptimizerWrapper.optimizeProgram(dmlProgram, runtimeProgram);
			} catch (DMLRuntimeException e) {
				throw new MLContextException("Exception occurred during global data flow optimization", e);
			} catch (HopsException e) {
				throw new MLContextException("Exception occurred during global data flow optimization", e);
			} catch (LopsException e) {
				throw new MLContextException("Exception occurred during global data flow optimization", e);
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
			Map<String, Object> inputParameters = script.getInputParameters();
			Map<String, String> inputParametersStringMaps = MLContextUtil
					.convertInputParametersForParser(inputParameters, script.getScriptType());

			String scriptExecutionString = script.getScriptExecutionString();
			dmlProgram = parser.parse(null, scriptExecutionString, inputParametersStringMaps);
		} catch (ParseException e) {
			throw new MLContextException("Exception occurred while parsing script", e);
		}
	}

	/**
	 * Replace persistent reads and writes with transient reads and writes in
	 * the symbol table.
	 */
	protected void rewritePersistentReadsAndWrites() {
		LocalVariableMap symbolTable = script.getSymbolTable();
		if (symbolTable != null) {
			String[] inputs = (script.getInputVariables() == null) ? new String[0]
					: script.getInputVariables().toArray(new String[0]);
			String[] outputs = (script.getOutputVariables() == null) ? new String[0]
					: script.getOutputVariables().toArray(new String[0]);
			RewriteRemovePersistentReadWrite rewrite = new RewriteRemovePersistentReadWrite(inputs, outputs,
					script.getSymbolTable());
			ProgramRewriter programRewriter = new ProgramRewriter(rewrite);
			try {
				programRewriter.rewriteProgramHopDAGs(dmlProgram);
			} catch (LanguageException | HopsException e) {
				throw new MLContextException("Exception occurred while rewriting persistent reads and writes", e);
			}
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
	 * Liveness analysis is performed on the program, obtaining sets of live-in
	 * and live-out variables by forward and backward passes over the program.
	 */
	protected void liveVariableAnalysis() {
		try {
			dmlTranslator = new DMLTranslator(dmlProgram);
			dmlTranslator.liveVariableAnalysis(dmlProgram);
		} catch (DMLRuntimeException e) {
			throw new MLContextException("Exception occurred during live variable analysis", e);
		} catch (LanguageException e) {
			throw new MLContextException("Exception occurred during live variable analysis", e);
		}
	}

	/**
	 * Semantically validate the program's expressions, statements, and
	 * statement blocks in a single recursive pass over the program. Constant
	 * and size propagation occurs during this step.
	 */
	protected void validateScript() {
		try {
			dmlTranslator.validateParseTree(dmlProgram);
		} catch (LanguageException e) {
			throw new MLContextException("Exception occurred while validating script", e);
		} catch (ParseException e) {
			throw new MLContextException("Exception occurred while validating script", e);
		} catch (IOException e) {
			throw new MLContextException("Exception occurred while validating script", e);
		}
	}

	/**
	 * Check that the Script object has a type (DML or PYDML) and a string
	 * representing the content of the Script.
	 */
	protected void checkScriptHasTypeAndString() {
		if (script == null) {
			throw new MLContextException("Script is null");
		} else if (script.getScriptType() == null) {
			throw new MLContextException("ScriptType (DML or PYDML) needs to be specified");
		} else if (script.getScriptString() == null) {
			throw new MLContextException("Script string is null");
		} else if (StringUtils.isBlank(script.getScriptString())) {
			throw new MLContextException("Script string is blank");
		}
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
	 * Obtain the translator
	 *
	 * @return the translator
	 */
	public DMLTranslator getDmlTranslator() {
		return dmlTranslator;
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
	public void setExplain(boolean explain) {
		this.explain = explain;
	}

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
			ExplainType explainType = explainLevel.getExplainType();
			DMLScript.EXPLAIN = explainType;
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
}
