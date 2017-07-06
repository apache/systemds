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
import java.util.Set;

import org.apache.log4j.Logger;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.SparkSession;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.api.jmlc.JMLCUtils;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.conf.DMLConfig;
import org.apache.sysml.parser.DataExpression;
import org.apache.sysml.parser.Expression;
import org.apache.sysml.parser.IntIdentifier;
import org.apache.sysml.parser.StringIdentifier;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.LocalVariableMap;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysml.runtime.instructions.Instruction;
import org.apache.sysml.runtime.instructions.cp.Data;
import org.apache.sysml.runtime.instructions.cp.ScalarObject;
import org.apache.sysml.runtime.matrix.MatrixFormatMetaData;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.utils.Explain.ExplainType;
import org.apache.sysml.utils.MLContextProxy;

/**
 * The MLContext API offers programmatic access to SystemML on Spark from
 * languages such as Scala, Java, and Python.
 *
 */
public class MLContext {
	/**
	 * Logger for MLContext
	 */
	public static Logger log = Logger.getLogger(MLContext.class);

	/**
	 * SparkSession object.
	 */
	private SparkSession spark = null;

	/**
	 * Reference to the current script.
	 */
	private Script executionScript = null;

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
	 * Whether or not GPU mode should be enabled
	 */
	private boolean gpu = false;

	/**
	 * Whether or not GPU mode should be force
	 */
	private boolean forceGPU = false;

	/**
	 * The number of heavy hitters that are printed as part of the statistics
	 * option
	 */
	private int statisticsMaxHeavyHitters = 10;

	/**
	 * The level and type of program explanation that should be displayed if
	 * explain is set to true.
	 */
	private ExplainLevel explainLevel = null;

	/**
	 * Whether or not all values should be maintained in the symbol table after
	 * execution.
	 */
	private boolean maintainSymbolTable = false;

	/**
	 * Whether or not the default ScriptExecutor should be initialized before
	 * execution. See {@link ScriptExecutor#init(boolean)}.
	 */
	private boolean initBeforeExecution = true;

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
	 * Create an MLContext based on a SparkSession for interaction with SystemML
	 * on Spark.
	 *
	 * @param spark
	 *            SparkSession
	 */
	public MLContext(SparkSession spark) {
		initMLContext(spark);
	}

	/**
	 * Create an MLContext based on a SparkContext for interaction with SystemML
	 * on Spark.
	 *
	 * @param sparkContext
	 *            SparkContext
	 */
	public MLContext(SparkContext sparkContext) {
		initMLContext(SparkSession.builder().sparkContext(sparkContext).getOrCreate());
	}

	/**
	 * Create an MLContext based on a JavaSparkContext for interaction with
	 * SystemML on Spark.
	 *
	 * @param javaSparkContext
	 *            JavaSparkContext
	 */
	public MLContext(JavaSparkContext javaSparkContext) {
		initMLContext(SparkSession.builder().sparkContext(javaSparkContext.sc()).getOrCreate());
	}

	/**
	 * Initialize MLContext. Verify Spark version supported, set default
	 * execution mode, set MLContextProxy, set default config, set compiler
	 * config.
	 *
	 * @param sc
	 *            SparkContext object.
	 */
	private void initMLContext(SparkSession spark) {

		try {
			MLContextUtil.verifySparkVersionSupported(spark);
		} catch (MLContextException e) {
			if (info() != null) {
				log.warn("Apache Spark " + this.info().minimumRecommendedSparkVersion()
						+ " or above is recommended for SystemML " + this.info().version());
			} else {
				try {
					String minSparkVersion = MLContextUtil.getMinimumRecommendedSparkVersionFromPom();
					log.warn("Apache Spark " + minSparkVersion
							+ " or above is recommended for this version of SystemML.");
				} catch (MLContextException e1) {
					log.error(
							"Minimum recommended Spark version could not be determined from SystemML jar file manifest or pom.xml");
				}
			}
		}

		if (activeMLContext == null) {
			System.out.println(MLContextUtil.welcomeMessage());
		}

		this.spark = spark;
		// by default, run in hybrid Spark mode for optimal performance
		DMLScript.rtplatform = RUNTIME_PLATFORM.HYBRID_SPARK;

		activeMLContext = this;
		MLContextProxy.setActive(true);

		MLContextUtil.setDefaultConfig();
		MLContextUtil.setCompilerConfig();
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
	 * @return the results as a MLResults object
	 */
	public MLResults execute(Script script) {
		ScriptExecutor scriptExecutor = new ScriptExecutor();
		scriptExecutor.setExplain(explain);
		scriptExecutor.setExplainLevel(explainLevel);
		scriptExecutor.setGPU(gpu);
		scriptExecutor.setForceGPU(forceGPU);
		scriptExecutor.setStatistics(statistics);
		scriptExecutor.setStatisticsMaxHeavyHitters(statisticsMaxHeavyHitters);
		scriptExecutor.setInit(initBeforeExecution);
		if (initBeforeExecution) {
			initBeforeExecution = false;
		}
		scriptExecutor.setMaintainSymbolTable(maintainSymbolTable);
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
	 * @return the results as a MLResults object
	 */
	public MLResults execute(Script script, ScriptExecutor scriptExecutor) {
		try {
			executionScript = script;

			Long time = new Long((new Date()).getTime());
			if ((script.getName() == null) || (script.getName().equals(""))) {
				script.setName(time.toString());
			}

			MLResults results = scriptExecutor.execute(script);

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
	 * Obtain the SparkSession associated with this MLContext.
	 *
	 * @return the SparkSession associated with this MLContext.
	 */
	public SparkSession getSparkSession() {
		return spark;
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
	 * Set the level of program explanation that should be displayed if explain
	 * is set to true.
	 *
	 * @param explainLevel
	 *            string denoting program explanation
	 */
	public void setExplainLevel(String explainLevel) {
		if (explainLevel != null) {
			for (ExplainLevel exp : ExplainLevel.values()) {
				String expString = exp.toString();
				if (expString.equalsIgnoreCase(explainLevel)) {
					setExplainLevel(exp);
					return;
				}
			}
		}
		throw new MLContextException("Failed to parse explain level: " + explainLevel + " "
				+ "(valid types: hops, runtime, recompile_hops, recompile_runtime).");
	}

	/**
	 * Whether or not to use (an available) GPU on the driver node. If a GPU is
	 * not available, and the GPU mode is set, SystemML will crash when the
	 * program is run.
	 *
	 * @param enable
	 *            true if needs to be enabled, false otherwise
	 */
	public void setGPU(boolean enable) {
		this.gpu = enable;
	}

	/**
	 * Whether or not to explicitly "force" the usage of GPU. If a GPU is not
	 * available, and the GPU mode is set or if available memory on GPU is less,
	 * SystemML will crash when the program is run.
	 *
	 * @param enable
	 *            true if needs to be enabled, false otherwise
	 */
	public void setForceGPU(boolean enable) {
		this.forceGPU = enable;
	}

	/**
	 * Whether or not the GPU mode is enabled.
	 *
	 * @return true if enabled, false otherwise
	 */
	public boolean isGPU() {
		return this.gpu;
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

				// Value retured from getVarParam is of type stringidentifier at
				// runtime, but at compile type its Expression
				// Could not find better way to compare this condition.
				Expression datatypeExp = ((DataExpression) source).getVarParam("data_type");
				String datatype = "matrix";
				if (datatypeExp != null)
					datatype = datatypeExp.toString();

				if (datatype.compareToIgnoreCase("frame") != 0) {
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
						exp.addVarParam(DataExpression.DATATYPEPARAM,
								new StringIdentifier("matrix", source.getFilename(), blp, bcp, elp, ecp));
						exp.addVarParam(DataExpression.VALUETYPEPARAM,
								new StringIdentifier("double", source.getFilename(), blp, bcp, elp, ecp));

						if (mo.getMetaData() instanceof MatrixFormatMetaData) {
							MatrixFormatMetaData metaData = (MatrixFormatMetaData) mo.getMetaData();
							if (metaData.getOutputInfo() == OutputInfo.CSVOutputInfo) {
								exp.addVarParam(DataExpression.FORMAT_TYPE,
										new StringIdentifier(DataExpression.FORMAT_TYPE_VALUE_CSV, source.getFilename(),
												blp, bcp, elp, ecp));
							} else if (metaData.getOutputInfo() == OutputInfo.TextCellOutputInfo) {
								exp.addVarParam(DataExpression.FORMAT_TYPE,
										new StringIdentifier(DataExpression.FORMAT_TYPE_VALUE_TEXT,
												source.getFilename(), blp, bcp, elp, ecp));
							} else if (metaData.getOutputInfo() == OutputInfo.BinaryBlockOutputInfo) {
								exp.addVarParam(DataExpression.ROWBLOCKCOUNTPARAM, new IntIdentifier(
										mo.getNumRowsPerBlock(), source.getFilename(), blp, bcp, elp, ecp));
								exp.addVarParam(DataExpression.COLUMNBLOCKCOUNTPARAM, new IntIdentifier(
										mo.getNumColumnsPerBlock(), source.getFilename(), blp, bcp, elp, ecp));
								exp.addVarParam(DataExpression.FORMAT_TYPE,
										new StringIdentifier(DataExpression.FORMAT_TYPE_VALUE_BINARY,
												source.getFilename(), blp, bcp, elp, ecp));
							} else {
								throw new MLContextException("Unsupported format through MLContext");
							}
						}
					}
				}
			}
		}

		private boolean isRegisteredAsInput(String parameterName) {
			if (executionScript != null) {
				Set<String> inputVariableNames = executionScript.getInputVariables();
				if (inputVariableNames != null) {
					return inputVariableNames.contains(parameterName);
				}
			}
			return false;
		}

		private MatrixObject getMatrixObject(String parameterName) {
			if (executionScript != null) {
				LocalVariableMap symbolTable = executionScript.getSymbolTable();
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
			if (executionScript == null || executionScript.getOutputVariables() == null)
				return instructions;

			Set<String> outputVariableNames = executionScript.getOutputVariables();
			return JMLCUtils.cleanupRuntimeInstructions(instructions, outputVariableNames.toArray(new String[0]));
		}
	}

	/**
	 * Used internally by MLContextProxy.
	 *
	 * @return InternalProxy object used by MLContextProxy
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
	 * Sets the maximum number of heavy hitters that are printed out as part of
	 * the statistics.
	 *
	 * @param maxHeavyHitters
	 *            maximum number of heavy hitters to print
	 */
	public void setStatisticsMaxHeavyHitters(int maxHeavyHitters) {
		DMLScript.STATISTICS_COUNT = maxHeavyHitters;
		this.statisticsMaxHeavyHitters = maxHeavyHitters;
	}

	/**
	 * Closes the mlcontext, which includes the cleanup of static and local
	 * state as well as scratch space and buffer pool cleanup. Note that the
	 * spark context is not explicitly closed to allow external reuse.
	 */
	public void close() {
		// reset static status (refs to sc / mlcontext)
		SparkExecutionContext.resetSparkContextStatic();
		MLContextProxy.setActive(false);
		activeMLContext = null;

		// cleanup scratch space and buffer pool
		try {
			DMLScript.cleanupHadoopExecution(ConfigurationManager.getDMLConfig());
		} catch (Exception ex) {
			throw new MLContextException("Failed to cleanup working directories.", ex);
		}

		// clear local status, but do not stop sc as it
		// may be used or stopped externally
		executionScript.clearAll();
		resetConfig();
		spark = null;
	}

	/**
	 * Obtain information about the project such as version and build time from
	 * the manifest in the SystemML jar file.
	 *
	 * @return information about the project
	 */
	public ProjectInfo info() {
		try {
			ProjectInfo projectInfo = ProjectInfo.getProjectInfo();
			return projectInfo;
		} catch (Exception e) {
			log.warn("Project information not available");
			return null;
		}
	}

	/**
	 * Obtain the SystemML version number.
	 *
	 * @return the SystemML version number
	 */
	public String version() {
		if (info() == null) {
			return "Version not available";
		}
		return info().version();
	}

	/**
	 * Obtain the SystemML jar file build time.
	 *
	 * @return the SystemML jar file build time
	 */
	public String buildTime() {
		if (info() == null) {
			return "Build time not available";
		}
		return info().buildTime();
	}

	/**
	 * Obtain the maximum number of heavy hitters that are printed out as part
	 * of the statistics.
	 *
	 * @return maximum number of heavy hitters to print
	 */
	public int getStatisticsMaxHeavyHitters() {
		return statisticsMaxHeavyHitters;
	}

	/**
	 * Whether or not the default ScriptExecutor should be initialized before
	 * execution.
	 *
	 * @return {@code true} if ScriptExecutor should be initialized before
	 *         execution, {@code false} otherwise
	 */
	public boolean isInitBeforeExecution() {
		return initBeforeExecution;
	}

	/**
	 * Whether or not the default ScriptExecutor should be initialized before
	 * execution.
	 *
	 * @param initBeforeExecution
	 *            {@code true} if ScriptExecutor should be initialized before
	 *            execution, {@code false} otherwise
	 */
	public void setInitBeforeExecution(boolean initBeforeExecution) {
		this.initBeforeExecution = initBeforeExecution;
	}

}
