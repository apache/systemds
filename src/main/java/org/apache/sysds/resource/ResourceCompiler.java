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
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.apache.sysds.api.DMLScript.*;

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
		dmlTranslator.constructHops(dmlProgram);
		dmlTranslator.rewriteHopsDAG(dmlProgram);
		dmlTranslator.constructLops(dmlProgram);
		dmlTranslator.rewriteLopDAG(dmlProgram);
		return dmlTranslator.getRuntimeProgram(dmlProgram, ConfigurationManager.getDMLConfig());
	}

	/**
	 * Recompiling a given program for resource optimization for single node execution
	 * @param program program to be recompiled
	 * @param driverMemory target driver memory
	 * @param driverCores target driver threads/cores
	 * @return the recompiled program as a new {@code Program} instance
	 */
	public static Program doFullRecompilation(Program program, long driverMemory, int driverCores) {
		setDriverConfigurations(driverMemory, driverCores);
		setSingleNodeExecution();
		return doFullRecompilation(program);
	}

	/**
	 * Recompiling a given program for resource optimization for Spark execution
	 * @param program program to be recompiled
	 * @param driverMemory target driver memory
	 * @param driverCores target driver threads/cores
	 * @param numberExecutors target number of executor nodes
	 * @param executorMemory target executor memory
	 * @param executorCores target executor threads/cores
	 * @return the recompiled program as a new {@code Program} instance
	 */
	public static Program doFullRecompilation(Program program, long driverMemory, int driverCores, int numberExecutors, long executorMemory, int executorCores) {
		setDriverConfigurations(driverMemory, driverCores);
		setExecutorConfigurations(numberExecutors, executorMemory, executorCores);
		return doFullRecompilation(program);
	}

	private static Program doFullRecompilation(Program program) {
		Program newProgram = new Program(program.getDMLProg());
		ArrayList<ProgramBlock> B = Stream.concat(
						program.getProgramBlocks().stream(),
						program.getFunctionProgramBlocks().values().stream())
				.collect(Collectors.toCollection(ArrayList::new));
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
	 * Sets resource configurations for the node executing the control program.
	 *
	 * @param nodeMemory memory in Bytes
	 * @param nodeNumCores number of CPU cores
	 */
	public static void setDriverConfigurations(long nodeMemory, int nodeNumCores) {
		// use 90% of the node's memory for the JVM heap -> rest needed for the OS
		InfrastructureAnalyzer.setLocalMaxMemory((long) (0.9 * nodeMemory));
		InfrastructureAnalyzer.setLocalPar(nodeNumCores);
	}

	/**
	 * Sets resource configurations for the cluster of nodes
	 * executing the Spark jobs.
	 *
	 * @param numExecutors number of nodes in cluster
	 * @param nodeMemory memory in Bytes per node
	 * @param nodeNumCores number of CPU cores per node
	 */
	public static void setExecutorConfigurations(int numExecutors, long nodeMemory, int nodeNumCores) {
		if (numExecutors > 0) {
			DMLScript.setGlobalExecMode(Types.ExecMode.HYBRID);
			SparkConf sparkConf = SparkExecutionContext.createSystemDSSparkConf();
			// ------------------ Static Configurations -------------------
			sparkConf.set("spark.master", "local[*]");
			sparkConf.set("spark.app.name", "SystemDS");
			sparkConf.set("spark.memory.useLegacyMode", "false");
			// ------------------ Static Configurations -------------------
			// ------------------ Dynamic Configurations -------------------
			sparkConf.set("spark.executor.memory", (nodeMemory/(1024*1024))+"m");
			sparkConf.set("spark.executor.instances", Integer.toString(numExecutors));
			sparkConf.set("spark.executor.cores", Integer.toString(nodeNumCores));
			// not setting "spark.default.parallelism" on purpose -> allows re-initialization
			// ------------------ Dynamic Configurations -------------------
			SparkExecutionContext.initLocalSparkContext(sparkConf);
		} else {
			throw new RuntimeException("The given number of executors was 0");
		}
	}

	public static void setSingleNodeExecution() {
		DMLScript.setGlobalExecMode(Types.ExecMode.SINGLE_NODE);
	}
}
