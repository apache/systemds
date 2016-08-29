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

import java.util.ArrayList;
import java.util.Date;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.api.MLContextProxy;
import org.apache.sysml.api.monitoring.SparkMonitoringUtil;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.conf.DMLConfig;
import org.apache.sysml.parser.DataExpression;
import org.apache.sysml.parser.Expression;
import org.apache.sysml.parser.IntIdentifier;
import org.apache.sysml.parser.StringIdentifier;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.LocalVariableMap;
import org.apache.sysml.runtime.controlprogram.caching.CacheableData;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysml.runtime.instructions.Instruction;
import org.apache.sysml.runtime.instructions.cp.Data;
import org.apache.sysml.runtime.instructions.cp.ScalarObject;
import org.apache.sysml.runtime.instructions.cp.VariableCPInstruction;
import org.apache.sysml.runtime.instructions.spark.functions.SparkListener;
import org.apache.sysml.runtime.matrix.MatrixFormatMetaData;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.utils.Explain.ExplainType;

/**
 * The MLContext API offers programmatic access to SystemML on Spark from
 * languages such as Scala, Java, and Python.
 *
 */
public class MLContext {
	/**
	 * Minimum Spark version supported by SystemML.
	 */
	public static final String SYSTEMML_MINIMUM_SPARK_VERSION = "1.4.0";

	/**
	 * SparkContext object.
	 */
	private SparkContext sc = null;

	/**
	 * SparkMonitoringUtil monitors SystemML performance on Spark.
	 */
	private SparkMonitoringUtil sparkMonitoringUtil = null;

	/**
	 * Reference to the currently executing script.
	 */
	private Script executingScript = null;

	/**
	 * The currently active MLContext.
	 */
	private static MLContext activeMLContext = null;

	/**
	 * Contains cleanup methods used by MLContextProxy.
	 */
	private InternalProxy internalProxy = new InternalProxy();

	/**
	 * Whether or not an explanation of the DML/PYDML program should be output
	 * to standard output.
	 */
	private boolean explain = false;

	/**
	 * Whether or not statistics of the DML/PYDML program execution should be
	 * output to standard output.
	 */
	private boolean statistics = false;

	/**
	 * The level and type of program explanation that should be displayed if
	 * explain is set to true.
	 */
	private ExplainLevel explainLevel = null;

	private List<String> scriptHistoryStrings = new ArrayList<String>();
	private Map<String, Script> scripts = new LinkedHashMap<String, Script>();

	/**
	 * The different explain levels supported by SystemML.
	 *
	 */
	public enum ExplainLevel {
		/** Explain disabled */
		NONE,
		/** Explain program and HOPs */
		HOPS,
		/** Explain runtime program */
		RUNTIME,
		/** Explain HOPs, including recompile */
		RECOMPILE_HOPS,
		/** Explain runtime program, including recompile */
		RECOMPILE_RUNTIME;

		public ExplainType getExplainType() {
			switch (this) {
			case NONE:
				return ExplainType.NONE;
			case HOPS:
				return ExplainType.HOPS;
			case RUNTIME:
				return ExplainType.RUNTIME;
			case RECOMPILE_HOPS:
				return ExplainType.RECOMPILE_HOPS;
			case RECOMPILE_RUNTIME:
				return ExplainType.RECOMPILE_RUNTIME;
			default:
				return ExplainType.HOPS;
			}
		}
	};

	/**
	 * Retrieve the currently active MLContext. This is used internally by
	 * SystemML via MLContextProxy.
	 * 
	 * @return the active MLContext
	 */
	public static MLContext getActiveMLContext() {
		return activeMLContext;
	}

	/**
	 * Create an MLContext based on a SparkContext for interaction with SystemML
	 * on Spark.
	 * 
	 * @param sparkContext
	 *            SparkContext
	 */
	public MLContext(SparkContext sparkContext) {
		this(sparkContext, false);
	}

	/**
	 * Create an MLContext based on a JavaSparkContext for interaction with
	 * SystemML on Spark.
	 * 
	 * @param javaSparkContext
	 *            JavaSparkContext
	 */
	public MLContext(JavaSparkContext javaSparkContext) {
		this(javaSparkContext.sc(), false);
	}

	/**
	 * Create an MLContext based on a SparkContext for interaction with SystemML
	 * on Spark, optionally monitor performance.
	 * 
	 * @param sc
	 *            SparkContext object.
	 * @param monitorPerformance
	 *            {@code true} if performance should be monitored, {@code false}
	 *            otherwise
	 */
	public MLContext(SparkContext sc, boolean monitorPerformance) {
		initMLContext(sc, monitorPerformance);
	}

	/**
	 * Initialize MLContext. Verify Spark version supported, set default
	 * execution mode, set MLContextProxy, set default config, set compiler
	 * config, and configure monitoring if needed.
	 * 
	 * @param sc
	 *            SparkContext object.
	 * @param monitorPerformance
	 *            {@code true} if performance should be monitored, {@code false}
	 *            otherwise
	 */
	private void initMLContext(SparkContext sc, boolean monitorPerformance) {

		if (activeMLContext == null) {
			System.out.println(MLContextUtil.welcomeMessage());
		}

		this.sc = sc;
		MLContextUtil.verifySparkVersionSupported(sc);
		// by default, run in hybrid Spark mode for optimal performance
		DMLScript.rtplatform = RUNTIME_PLATFORM.HYBRID_SPARK;

		activeMLContext = this;
		MLContextProxy.setActive(true);

		MLContextUtil.setDefaultConfig();
		MLContextUtil.setCompilerConfig();

		if (monitorPerformance) {
			SparkListener sparkListener = new SparkListener(sc);
			sparkMonitoringUtil = new SparkMonitoringUtil(sparkListener);
			sc.addSparkListener(sparkListener);
		}
	}

	/**
	 * Clean up the variables from the buffer pool, including evicted files,
	 * because the buffer pool holds references.
	 */
	public void clearCache() {
		CacheableData.cleanupCacheDir();
	}

	/**
	 * Reset configuration settings to default settings.
	 */
	public void resetConfig() {
		MLContextUtil.setDefaultConfig();
	}

	/**
	 * Set configuration property, such as
	 * {@code setConfigProperty("localtmpdir", "/tmp/systemml")}.
	 * 
	 * @param propertyName
	 *            property name
	 * @param propertyValue
	 *            property value
	 */
	public void setConfigProperty(String propertyName, String propertyValue) {
		DMLConfig config = ConfigurationManager.getDMLConfig();
		try {
			config.setTextValue(propertyName, propertyValue);
		} catch (DMLRuntimeException e) {
			throw new MLContextException(e);
		}
	}

	/**
	 * Execute a DML or PYDML Script.
	 * 
	 * @param script
	 *            The DML or PYDML Script object to execute.
	 */
	public MLResults execute(Script script) {
		ScriptExecutor scriptExecutor = new ScriptExecutor(sparkMonitoringUtil);
		scriptExecutor.setExplain(explain);
		scriptExecutor.setExplainLevel(explainLevel);
		scriptExecutor.setStatistics(statistics);
		return execute(script, scriptExecutor);
	}

	/**
	 * Execute a DML or PYDML Script object using a ScriptExecutor. The
	 * ScriptExecutor class can be extended to allow the modification of the
	 * default execution pathway.
	 * 
	 * @param script
	 *            the DML or PYDML Script object
	 * @param scriptExecutor
	 *            the ScriptExecutor that defines the script execution pathway
	 */
	public MLResults execute(Script script, ScriptExecutor scriptExecutor) {
		try {
			executingScript = script;

			Long time = new Long((new Date()).getTime());
			if ((script.getName() == null) || (script.getName().equals(""))) {
				script.setName(time.toString());
			}

			MLResults results = scriptExecutor.execute(script);

			String history = MLContextUtil.createHistoryForScript(script, time);
			scriptHistoryStrings.add(history);
			scripts.put(script.getName(), script);

			return results;
		} catch (RuntimeException e) {
			throw new MLContextException("Exception when executing script", e);
		}
	}

	/**
	 * Set SystemML configuration based on a configuration file.
	 * 
	 * @param configFilePath
	 *            path to the configuration file
	 */
	public void setConfig(String configFilePath) {
		MLContextUtil.setConfig(configFilePath);
	}

	/**
	 * Obtain the SparkMonitoringUtil if it is available.
	 * 
	 * @return the SparkMonitoringUtil if it is available.
	 */
	public SparkMonitoringUtil getSparkMonitoringUtil() {
		return sparkMonitoringUtil;
	}

	/**
	 * Obtain the SparkContext associated with this MLContext.
	 * 
	 * @return the SparkContext associated with this MLContext.
	 */
	public SparkContext getSparkContext() {
		return sc;
	}

	/**
	 * Whether or not an explanation of the DML/PYDML program should be output
	 * to standard output.
	 * 
	 * @return {@code true} if explanation should be output, {@code false}
	 *         otherwise
	 */
	public boolean isExplain() {
		return explain;
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
	 * Set the level of program explanation that should be displayed if explain
	 * is set to true.
	 * 
	 * @param explainLevel
	 *            the level of program explanation
	 */
	public void setExplainLevel(ExplainLevel explainLevel) {
		this.explainLevel = explainLevel;
	}

	/**
	 * Used internally by MLContextProxy.
	 *
	 */
	public class InternalProxy {

		public void setAppropriateVarsForRead(Expression source, String target) {
			boolean isTargetRegistered = isRegisteredAsInput(target);
			boolean isReadExpression = (source instanceof DataExpression && ((DataExpression) source).isRead());
			if (isTargetRegistered && isReadExpression) {
				DataExpression exp = (DataExpression) source;
				// Do not check metadata file for registered reads
				exp.setCheckMetadata(false);
				
				//Value retured from getVarParam is of type stringidentifier at runtime, but at compile type its Expression
				//Could not find better way to compare this condition.
				Expression datatypeExp = ((DataExpression)source).getVarParam("data_type");
				String datatype = "matrix";
				if(datatypeExp != null)
					datatype = datatypeExp.toString();

				if(datatype.compareToIgnoreCase("frame") != 0) {
					MatrixObject mo = getMatrixObject(target);
					if (mo != null) {
						int blp = source.getBeginLine();
						int bcp = source.getBeginColumn();
						int elp = source.getEndLine();
						int ecp = source.getEndColumn();
						exp.addVarParam(DataExpression.READROWPARAM,
								new IntIdentifier(mo.getNumRows(), source.getFilename(), blp, bcp, elp, ecp));
						exp.addVarParam(DataExpression.READCOLPARAM,
								new IntIdentifier(mo.getNumColumns(), source.getFilename(), blp, bcp, elp, ecp));
						exp.addVarParam(DataExpression.READNUMNONZEROPARAM,
								new IntIdentifier(mo.getNnz(), source.getFilename(), blp, bcp, elp, ecp));
						exp.addVarParam(DataExpression.DATATYPEPARAM, new StringIdentifier("matrix", source.getFilename(),
								blp, bcp, elp, ecp));
						exp.addVarParam(DataExpression.VALUETYPEPARAM, new StringIdentifier("double", source.getFilename(),
								blp, bcp, elp, ecp));
	
						if (mo.getMetaData() instanceof MatrixFormatMetaData) {
							MatrixFormatMetaData metaData = (MatrixFormatMetaData) mo.getMetaData();
							if (metaData.getOutputInfo() == OutputInfo.CSVOutputInfo) {
								exp.addVarParam(DataExpression.FORMAT_TYPE, new StringIdentifier(
										DataExpression.FORMAT_TYPE_VALUE_CSV, source.getFilename(), blp, bcp, elp, ecp));
							} else if (metaData.getOutputInfo() == OutputInfo.TextCellOutputInfo) {
								exp.addVarParam(DataExpression.FORMAT_TYPE, new StringIdentifier(
										DataExpression.FORMAT_TYPE_VALUE_TEXT, source.getFilename(), blp, bcp, elp, ecp));
							} else if (metaData.getOutputInfo() == OutputInfo.BinaryBlockOutputInfo) {
								exp.addVarParam(
										DataExpression.ROWBLOCKCOUNTPARAM,
										new IntIdentifier(mo.getNumRowsPerBlock(), source.getFilename(), blp, bcp, elp, ecp));
								exp.addVarParam(DataExpression.COLUMNBLOCKCOUNTPARAM,
										new IntIdentifier(mo.getNumColumnsPerBlock(), source.getFilename(), blp, bcp, elp,
												ecp));
								exp.addVarParam(DataExpression.FORMAT_TYPE, new StringIdentifier(
										DataExpression.FORMAT_TYPE_VALUE_BINARY, source.getFilename(), blp, bcp, elp, ecp));
							} else {
								throw new MLContextException("Unsupported format through MLContext");
							}
						}
					}
				}
			}
		}

		private boolean isRegisteredAsInput(String parameterName) {
			if (executingScript != null) {
				Set<String> inputVariableNames = executingScript.getInputVariables();
				if (inputVariableNames != null) {
					return inputVariableNames.contains(parameterName);
				}
			}
			return false;
		}

		private MatrixObject getMatrixObject(String parameterName) {
			if (executingScript != null) {
				LocalVariableMap symbolTable = executingScript.getSymbolTable();
				if (symbolTable != null) {
					Data data = symbolTable.get(parameterName);
					if (data instanceof MatrixObject) {
						return (MatrixObject) data;
					} else {
						if (data instanceof ScalarObject) {
							return null;
						}
					}
				}
			}
			throw new MLContextException("getMatrixObject not set for parameter: " + parameterName);
		}

		public ArrayList<Instruction> performCleanupAfterRecompilation(ArrayList<Instruction> instructions) {
			if (executingScript == null) {
				return instructions;
			}
			Set<String> outputVariableNames = executingScript.getOutputVariables();
			if (outputVariableNames == null) {
				return instructions;
			}

			for (int i = 0; i < instructions.size(); i++) {
				Instruction inst = instructions.get(i);
				if (inst instanceof VariableCPInstruction && ((VariableCPInstruction) inst).isRemoveVariable()) {
					VariableCPInstruction varInst = (VariableCPInstruction) inst;
					for (String outputVariableName : outputVariableNames)
						if (varInst.isRemoveVariable(outputVariableName)) {
							instructions.remove(i);
							i--;
							break;
						}
				}
			}
			return instructions;
		}
	}

	/**
	 * Used internally by MLContextProxy.
	 *
	 */
	public InternalProxy getInternalProxy() {
		return internalProxy;
	}

	/**
	 * Whether or not statistics of the DML/PYDML program execution should be
	 * output to standard output.
	 * 
	 * @return {@code true} if statistics should be output, {@code false}
	 *         otherwise
	 */
	public boolean isStatistics() {
		return statistics;
	}

	/**
	 * Whether or not statistics of the DML/PYDML program execution should be
	 * output to standard output.
	 * 
	 * @param statistics
	 *            {@code true} if statistics should be output, {@code false}
	 *            otherwise
	 */
	public void setStatistics(boolean statistics) {
		DMLScript.STATISTICS = statistics;
		this.statistics = statistics;
	}

	/**
	 * Obtain a map of the scripts that have executed.
	 * 
	 * @return a map of the scripts that have executed
	 */
	public Map<String, Script> getScripts() {
		return scripts;
	}

	/**
	 * Obtain a script that has executed by name.
	 * 
	 * @param name
	 *            the name of the script
	 * @return the script corresponding to the name
	 */
	public Script getScriptByName(String name) {
		Script script = scripts.get(name);
		if (script == null) {
			throw new MLContextException("Script with name '" + name + "' not found.");
		}
		return script;
	}

	/**
	 * Display the history of scripts that have executed.
	 * 
	 * @return the history of scripts that have executed
	 */
	public String history() {
		return MLContextUtil.displayScriptHistory(scriptHistoryStrings);
	}

	/**
	 * Clear all the scripts, removing them from the history, and clear the
	 * cache.
	 */
	public void clear() {
		Set<String> scriptNames = scripts.keySet();
		for (String scriptName : scriptNames) {
			Script script = scripts.get(scriptName);
			script.clearAll();
		}

		scripts.clear();
		scriptHistoryStrings.clear();

		clearCache();
	}

	public void close() {
		// reset static status (refs to sc / mlcontext)
		SparkExecutionContext.resetSparkContextStatic();
		MLContextProxy.setActive(false);
		activeMLContext = null;

		// clear local status, but do not stop sc as it
		// may be used or stopped externally
		clear();
		resetConfig();
		sc = null;
	}
}
