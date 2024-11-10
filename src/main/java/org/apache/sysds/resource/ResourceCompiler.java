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

package org.apache.sysds.resource;

import org.apache.spark.SparkConf;
import org.apache.sysds.api.DMLOptions;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.hops.recompile.Recompiler;
import org.apache.sysds.parser.*;
import org.apache.sysds.runtime.controlprogram.*;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContextFactory;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.utils.stats.InfrastructureAnalyzer;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.apache.sysds.api.DMLScript.*;
import static org.apache.sysds.parser.DataExpression.IO_FILENAME;
import static org.apache.sysds.resource.CloudUtils.*;

/**
 * This class does full or partial program recompilation
 * based on given runtime program. It uses the methods provided
 * by {@code hops.recompile.Recompiler}).
 * It keeps a state of the current recompilation phase in order
 * to decide when to do full recompilation and when not.
 */
public class ResourceCompiler {
	public static final long DEFAULT_DRIVER_MEMORY = 512*1024*1024; // 0.5GB
	public static final int DEFAULT_DRIVER_THREADS = 1; // 0.5GB
	public static final long DEFAULT_EXECUTOR_MEMORY = 512*1024*1024; // 0.5GB
	public static final int DEFAULT_EXECUTOR_THREADS = 2; // avoids creating spark context
	public static final int DEFAULT_NUMBER_EXECUTORS = 2; // avoids creating spark context

	public static Program compile(String filePath, Map<String, String> args) throws IOException {
		return compile(filePath, args, null);
	}

	public static Program compile(String filePath, Map<String, String> args, HashMap<String, String> replaceVars) throws IOException {
		// setting the dynamic recompilation flags during resource optimization is obsolete
		DMLOptions dmlOptions =DMLOptions.defaultOptions;
		dmlOptions.argVals = args;

		String dmlScriptStr = readDMLScript(true, filePath);
		Map<String, String> argVals = dmlOptions.argVals;

		ParserWrapper parser = ParserFactory.createParser();
		DMLProgram dmlProgram = parser.parse(null, dmlScriptStr, argVals);
		DMLTranslator dmlTranslator = new DMLTranslator(dmlProgram);
		dmlTranslator.liveVariableAnalysis(dmlProgram);
		dmlTranslator.validateParseTree(dmlProgram);
		if (replaceVars != null && !replaceVars.isEmpty()) {
			replaceFilename(dmlProgram, replaceVars);}
		dmlTranslator.constructHops(dmlProgram);
		dmlTranslator.rewriteHopsDAG(dmlProgram);
		dmlTranslator.constructLops(dmlProgram);
		dmlTranslator.rewriteLopDAG(dmlProgram);
		return dmlTranslator.getRuntimeProgram(dmlProgram, ConfigurationManager.getDMLConfig());
	}

	public static void replaceFilename(DMLProgram dmlp, HashMap<String, String> replaceVars)
	{
		for (int i = 0; i < dmlp.getNumStatementBlocks(); i++) {
			StatementBlock sb = dmlp.getStatementBlock(i);
			for (Statement statement: sb.getStatements()) {
				if (!(statement instanceof AssignmentStatement ||
						statement instanceof OutputStatement)) continue;

				StringIdentifier stringIdentifier;
				if (statement instanceof AssignmentStatement) {
					Expression assignExpression = ((AssignmentStatement) statement).getSource();
					if (!(assignExpression instanceof StringIdentifier ||
							assignExpression instanceof DataExpression)) continue;

					if (assignExpression instanceof DataExpression) {
						Expression filenameExpression = ((DataExpression) assignExpression).getVarParam(IO_FILENAME);
						if (!(filenameExpression instanceof StringIdentifier)) continue;

						stringIdentifier = (StringIdentifier) filenameExpression;
					} else {
						stringIdentifier = (StringIdentifier) assignExpression;
					}
				} else {
					Expression filenameExpression = ((OutputStatement) statement).getExprParam(IO_FILENAME);
					if (!(filenameExpression instanceof StringIdentifier)) continue;

					stringIdentifier = (StringIdentifier) filenameExpression;
				}

				if (!(replaceVars.containsKey(stringIdentifier.getValue()))) continue;
				String valToReplace = replaceVars.get(stringIdentifier.getValue());
				stringIdentifier.setValue(valToReplace);
			}
		}
	}

	/**
	 * Recompiling a given program for resource optimization.
	 * This method should always be called after setting the target resources
	 * to {@code InfrastructureAnalyzer} and {@code SparkExecutionContext}
	 *
	 * @param program program to be recompiled
	 * @return the recompiled program as a new {@code Program} instance
	 */
	public static Program doFullRecompilation(Program program) {
		// adjust defaults for memory estimates of variables with unknown dimensions
		OptimizerUtils.resetDefaultSize();
		// init new Program object for the output
		Program newProgram = new Program(program.getDMLProg());
		// collect program blocks from all layers
		ArrayList<ProgramBlock> B = Stream.concat(
						program.getProgramBlocks().stream(),
						program.getFunctionProgramBlocks().values().stream())
				.collect(Collectors.toCollection(ArrayList::new));
		// recompile for each fo the program blocks and put attach the program object
		doRecompilation(B, newProgram);
		return newProgram;
	}

	private static void doRecompilation(ArrayList<ProgramBlock> origin, Program target) {
		for (ProgramBlock originBlock : origin) {
			doRecompilation(originBlock, target);
		}
	}

	private static void doRecompilation(ProgramBlock originBlock, Program target) {
		if (originBlock instanceof FunctionProgramBlock)
		{
			FunctionProgramBlock fpb = (FunctionProgramBlock)originBlock;
			Recompiler.recompileProgramBlockHierarchy(fpb.getChildBlocks(), new LocalVariableMap(), 0, true, Recompiler.ResetType.NO_RESET);
			String functionName = ((FunctionStatement) fpb.getStatementBlock().getStatement(0)).getName();
			String namespace = null;
			for (Map.Entry<String, FunctionDictionary<FunctionStatementBlock>> pairNS: target.getDMLProg().getNamespaces().entrySet()) {
				if (pairNS.getValue().containsFunction(functionName)) {
					namespace = pairNS.getKey();
				}
			}
			target.addFunctionProgramBlock(namespace, functionName, fpb);
		}
		else if (originBlock instanceof IfProgramBlock)
		{
			IfProgramBlock ipb = (IfProgramBlock)originBlock;
			IfStatementBlock sb = (IfStatementBlock) ipb.getStatementBlock();
			if(sb!=null && sb.getPredicateHops()!=null ){
				ArrayList<Hop> hopAsList = new ArrayList<>(Collections.singletonList(sb.getPredicateHops()));
				ArrayList<Instruction> inst = Recompiler.recompile(null , hopAsList, null, null, true, false, true, false, false, null, 0);
				ipb.setPredicate(inst);
				target.addProgramBlock(ipb);
			}
			doRecompilation(ipb.getChildBlocksIfBody(), target);
			doRecompilation(ipb.getChildBlocksElseBody(), target);
		}
		else if (originBlock instanceof WhileProgramBlock)
		{
			WhileProgramBlock wpb = (WhileProgramBlock)originBlock;
			WhileStatementBlock sb = (WhileStatementBlock) originBlock.getStatementBlock();
			if(sb!=null && sb.getPredicateHops()!=null ){
				ArrayList<Hop> hopAsList = new ArrayList<>(Collections.singletonList(sb.getPredicateHops()));
				ArrayList<Instruction> inst = Recompiler.recompile(null , hopAsList, null, null, true, false, true, false, false, null, 0);
				wpb.setPredicate(inst);
				target.addProgramBlock(wpb);
			}
			doRecompilation(wpb.getChildBlocks(), target);
		}
		else if (originBlock instanceof ForProgramBlock) //incl parfor
		{
			ForProgramBlock fpb = (ForProgramBlock)originBlock;
			ForStatementBlock sb = (ForStatementBlock) fpb.getStatementBlock();
			if(sb!=null){
				if( sb.getFromHops()!=null ){
					ArrayList<Hop> hopAsList = new ArrayList<>(Collections.singletonList(sb.getFromHops()));
					ArrayList<Instruction> inst = Recompiler.recompile(null , hopAsList, null, null, true, false, true, false, false, null, 0);
					fpb.setFromInstructions( inst );
				}
				if(sb.getToHops()!=null){
					ArrayList<Hop> hopAsList = new ArrayList<>(Collections.singletonList(sb.getToHops()));
					ArrayList<Instruction> inst = Recompiler.recompile(null , hopAsList, null, null, true, false, true, false, false, null, 0);
					fpb.setToInstructions( inst );
				}
				if(sb.getIncrementHops()!=null){
					ArrayList<Hop> hopAsList = new ArrayList<>(Collections.singletonList(sb.getIncrementHops()));
					ArrayList<Instruction> inst = Recompiler.recompile(null , hopAsList, null, null, true, false, true, false, false, null, 0);
					fpb.setIncrementInstructions(inst);
				}
				target.addProgramBlock(fpb);

			}
			doRecompilation(fpb.getChildBlocks(), target);
		}
		else
		{
			BasicProgramBlock bpb = (BasicProgramBlock)originBlock;
			StatementBlock sb = bpb.getStatementBlock();
			ArrayList<Instruction> inst = Recompiler.recompile(sb, sb.getHops(), ExecutionContextFactory.createContext(target), null, true, false, true, false, false, null, 0);
			bpb.setInstructions(inst);
			target.addProgramBlock(bpb);
		}
	}

	/**
	 * Sets resource configurations for executions in single-node mode
	 * including the hardware configurations for the node running the CP.
	 *
	 * @param nodeMemory memory budget for the node running CP
	 * @param nodeCores number of CPU cores for the node running CP
	 */
	public static void setSingleNodeResourceConfigs(long nodeMemory, int nodeCores) {
		DMLScript.setGlobalExecMode(Types.ExecMode.SINGLE_NODE);
		// use 90% of the node's memory for the JVM heap -> rest needed for the OS
		long effectiveSingleNodeMemory = (long) (nodeMemory * JVM_MEMORY_FACTOR);
		// CPU core would be shared with OS -> no further limitation
		InfrastructureAnalyzer.setLocalMaxMemory(effectiveSingleNodeMemory);
		InfrastructureAnalyzer.setLocalPar(nodeCores);
	}

	/**
	 * Sets resource configurations for  executions in hybrid mode
	 * including the hardware configurations for the node running the CP
	 * and the worker nodes running Spark executors
	 *
	 * @param driverMemory memory budget for the node running CP
	 * @param driverCores number of CPU cores for the node running CP
	 * @param numExecutors   number of nodes in cluster
	 * @param executorMemory memory budget for the nodes running executors
	 * @param executorCores  number of CPU cores for the nodes running executors
	 */
	public static void setSparkClusterResourceConfigs(long driverMemory, int driverCores, int numExecutors, long executorMemory, int executorCores) {
		if (numExecutors <= 0) {
			throw new RuntimeException("The given number of executors was non-positive");
		}
		// ------------------- CP (driver) configurations -------------------
		// use at most 90% of the node's memory for the JVM heap -> rest needed for the OS and resource management
		// adapt the minimum based on the need for YAN RM
		long effectiveDriverMemory = calculateEffectiveDriverMemoryBudget(driverMemory, numExecutors*executorCores);
		// require that always at least half of the memory budget is left for driver memory or 1GB
		if (effectiveDriverMemory <= GBtoBytes(1)  || driverMemory > 2*effectiveDriverMemory) {
			throw new IllegalArgumentException("Driver resources are not sufficient to handle the cluster");
		}
		// CPU core would be shared -> no further limitation
		InfrastructureAnalyzer.setLocalMaxMemory(effectiveDriverMemory);
		InfrastructureAnalyzer.setLocalPar(driverCores);

		// ---------------------- Spark Configurations -----------------------
		DMLScript.setGlobalExecMode(Types.ExecMode.HYBRID);
		SparkConf sparkConf = SparkExecutionContext.createSystemDSSparkConf();

		// ------------------ Static Spark Configurations --------------------
		sparkConf.set("spark.master", "local[*]");
		sparkConf.set("spark.app.name", "SystemDS");
		sparkConf.set("spark.memory.useLegacyMode", "false");

		// ------------------ Dynamic Spark Configurations -------------------
		// calculate the effective resource that would be available for the executor containers in YARN
		int[] effectiveValues = getEffectiveExecutorResources(executorMemory, executorCores, numExecutors);
		int effectiveExecutorMemory = effectiveValues[0];
		int effectiveExecutorCores = effectiveValues[1];
		int effectiveNumExecutor = effectiveValues[2];
		sparkConf.set("spark.executor.memory", (effectiveExecutorMemory)+"m");
		sparkConf.set("spark.executor.instances", Integer.toString(effectiveNumExecutor));
		sparkConf.set("spark.executor.cores", Integer.toString(effectiveExecutorCores));
		// not setting "spark.default.parallelism" on purpose -> allows re-initialization

		// ------------------- Load Spark Configurations ---------------------
		SparkExecutionContext.initLocalSparkContext(sparkConf);
	}
}
