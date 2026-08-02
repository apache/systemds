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

package org.apache.sysds.test.component.compile;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.hops.recompile.Recompiler;
import org.apache.sysds.parser.DMLProgram;
import org.apache.sysds.parser.DMLTranslator;
import org.apache.sysds.parser.ParserFactory;
import org.apache.sysds.parser.ParserWrapper;
import org.apache.sysds.runtime.controlprogram.BasicProgramBlock;
import org.apache.sysds.runtime.controlprogram.ForProgramBlock;
import org.apache.sysds.runtime.controlprogram.FunctionProgramBlock;
import org.apache.sysds.runtime.controlprogram.IfProgramBlock;
import org.apache.sysds.runtime.controlprogram.Program;
import org.apache.sysds.runtime.controlprogram.ProgramBlock;
import org.apache.sysds.runtime.controlprogram.WhileProgramBlock;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.instructions.cp.CPInstruction;
import org.apache.sysds.runtime.instructions.spark.SPInstruction;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.utils.Explain;
import org.apache.sysds.utils.stats.InfrastructureAnalyzer;
import org.junit.Assert;

/**
 * Base class for compilation-verification tests: compile a DML script into a runtime {@link Program} and inspect the
 * resulting plan (instructions and their exec types) without ever executing it.
 */
public abstract class CompilerTestBase extends AutomatedTestBase {

	/** A small default local memory budget (8 MB) that forces large operations onto Spark in HYBRID mode. */
	public static final long SMALL_MEM_BUDGET = 8L * 1024 * 1024;

	@Override
	public void setUp() {
		// no test-configuration setup needed; scripts are compiled from in-memory strings
	}

	/**
	 * Compile a DML script string into a runtime {@link Program} without executing it.
	 *
	 * @param dmlScript    the DML source
	 * @param args         named command-line arguments ($name -&gt; value), may be null
	 * @param mode         the global execution mode (e.g. {@link ExecMode#HYBRID})
	 * @param localMaxMem  the local memory budget in bytes used for memory-based exec-type decisions
	 * @return the compiled runtime program
	 */
	protected Program compile(String dmlScript, Map<String, String> args, ExecMode mode, long localMaxMem) {
		final ExecMode oldMode = DMLScript.getGlobalExecMode();
		final long oldMem = InfrastructureAnalyzer.getLocalMaxMemory();
		final DMLConfig oldConfig = ConfigurationManager.getDMLConfig();
		try {
			ConfigurationManager.setGlobalConfig(new DMLConfig());
			DMLScript.setGlobalExecMode(mode);
			InfrastructureAnalyzer.setLocalMaxMemory(localMaxMem);
			OptimizerUtils.resetDefaultSize();

			Map<String, String> argVals = (args == null) ? new HashMap<>() : new HashMap<>(args);
			ParserWrapper parser = ParserFactory.createParser();
			DMLProgram prog = parser.parse(null, dmlScript, argVals);
			DMLTranslator dmlt = new DMLTranslator(prog);
			dmlt.liveVariableAnalysis(prog);
			dmlt.validateParseTree(prog);
			dmlt.constructHops(prog);
			dmlt.rewriteHopsDAG(prog);
			dmlt.constructLops(prog);
			dmlt.rewriteLopDAG(prog);
			return dmlt.getRuntimeProgram(prog, ConfigurationManager.getDMLConfig());
		}
		catch(Exception e) {
			throw new RuntimeException("Failed to compile DML script:\n" + dmlScript, e);
		}
		finally {
			DMLScript.setGlobalExecMode(oldMode);
			InfrastructureAnalyzer.setLocalMaxMemory(oldMem);
			ConfigurationManager.setGlobalConfig(oldConfig);
			Recompiler.reinitRecompiler();
		}
	}

	/** Recursively collect every instruction in the program, including control-flow predicates and function bodies. */
	protected List<Instruction> getInstructions(Program prog) {
		List<Instruction> out = new ArrayList<>();
		for(ProgramBlock pb : prog.getProgramBlocks())
			collect(pb, out);
		for(FunctionProgramBlock fpb : prog.getFunctionProgramBlocks(false).values())
			collect(fpb, out);
		return out;
	}

	private void collect(ProgramBlock pb, List<Instruction> out) {
		if(pb instanceof BasicProgramBlock) {
			out.addAll(((BasicProgramBlock) pb).getInstructions());
		}
		else if(pb instanceof IfProgramBlock) {
			IfProgramBlock ipb = (IfProgramBlock) pb;
			out.addAll(ipb.getPredicate());
			ipb.getChildBlocksIfBody().forEach(c -> collect(c, out));
			ipb.getChildBlocksElseBody().forEach(c -> collect(c, out));
		}
		else if(pb instanceof WhileProgramBlock) {
			WhileProgramBlock wpb = (WhileProgramBlock) pb;
			out.addAll(wpb.getPredicate());
			wpb.getChildBlocks().forEach(c -> collect(c, out));
		}
		else if(pb instanceof ForProgramBlock) { // incl. ParForProgramBlock
			ForProgramBlock fpb = (ForProgramBlock) pb;
			out.addAll(fpb.getFromInstructions());
			out.addAll(fpb.getToInstructions());
			out.addAll(fpb.getIncrementInstructions());
			fpb.getChildBlocks().forEach(c -> collect(c, out));
		}
		else if(pb instanceof FunctionProgramBlock) {
			((FunctionProgramBlock) pb).getChildBlocks().forEach(c -> collect(c, out));
		}
	}

	/** All instructions whose opcode equals {@code opcode} (exact match). */
	protected List<Instruction> getByOpcode(Program prog, String opcode) {
		return getInstructions(prog).stream().filter(i -> opcode.equals(i.getOpcode()))
			.collect(Collectors.toList());
	}

	protected static boolean isSpark(Instruction inst) {
		return inst instanceof SPInstruction;
	}

	protected static boolean isCP(Instruction inst) {
		return inst instanceof CPInstruction;
	}

	/** Assert that at least one instruction with the given opcode exists and that all such instructions are Spark. */
	protected void assertSpark(Program prog, String opcode) {
		assertExecType(prog, opcode, true);
	}

	/** Assert that at least one instruction with the given opcode exists and that all such instructions are CP. */
	protected void assertCP(Program prog, String opcode) {
		assertExecType(prog, opcode, false);
	}

	private void assertExecType(Program prog, String opcode, boolean expectSpark) {
		List<Instruction> matches = getByOpcode(prog, opcode);
		Assert.assertFalse("Expected at least one '" + opcode + "' instruction but found none.\n"
			+ Explain.explain(prog), matches.isEmpty());
		for(Instruction inst : matches) {
			boolean spark = isSpark(inst);
			Assert.assertEquals("Instruction '" + opcode + "' expected exec type "
				+ (expectSpark ? "SPARK" : "CP") + " but was " + (spark ? "SPARK" : "CP") + ".\n"
				+ Explain.explain(prog), expectSpark, spark);
		}
	}

	protected long countSpark(Program prog) {
		return getInstructions(prog).stream().filter(CompilerTestBase::isSpark).count();
	}

	protected String explain(Program prog) {
		return Explain.explain(prog);
	}
}
