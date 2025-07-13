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

package org.apache.sysds.utils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashSet;
import java.util.Map;
import java.util.List;
import java.util.Map.Entry;
import java.util.Set;
import java.util.Stack;

import org.apache.commons.lang3.mutable.MutableInt;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.LiteralOp;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.hops.codegen.cplan.CNode;
import org.apache.sysds.hops.codegen.cplan.CNodeMultiAgg;
import org.apache.sysds.hops.codegen.cplan.CNodeTpl;
import org.apache.sysds.hops.ipa.FunctionCallGraph;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.parser.DMLProgram;
import org.apache.sysds.parser.ForStatement;
import org.apache.sysds.parser.ForStatementBlock;
import org.apache.sysds.parser.FunctionStatement;
import org.apache.sysds.parser.FunctionStatementBlock;
import org.apache.sysds.parser.IfStatement;
import org.apache.sysds.parser.IfStatementBlock;
import org.apache.sysds.parser.ParForStatementBlock;
import org.apache.sysds.parser.StatementBlock;
import org.apache.sysds.parser.WhileStatement;
import org.apache.sysds.parser.WhileStatementBlock;
import org.apache.sysds.runtime.controlprogram.BasicProgramBlock;
import org.apache.sysds.runtime.controlprogram.ForProgramBlock;
import org.apache.sysds.runtime.controlprogram.FunctionProgramBlock;
import org.apache.sysds.runtime.controlprogram.IfProgramBlock;
import org.apache.sysds.runtime.controlprogram.ParForProgramBlock;
import org.apache.sysds.runtime.controlprogram.Program;
import org.apache.sysds.runtime.controlprogram.ProgramBlock;
import org.apache.sysds.runtime.controlprogram.WhileProgramBlock;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.instructions.cp.CPInstruction;
import org.apache.sysds.runtime.instructions.fed.FEDInstruction;
import org.apache.sysds.runtime.instructions.gpu.GPUInstruction;
import org.apache.sysds.runtime.instructions.ooc.OOCInstruction;
import org.apache.sysds.runtime.instructions.spark.CSVReblockSPInstruction;
import org.apache.sysds.runtime.instructions.spark.CheckpointSPInstruction;
import org.apache.sysds.runtime.instructions.spark.ReblockSPInstruction;
import org.apache.sysds.runtime.instructions.spark.SPInstruction;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.lineage.LineageItemUtils;
import org.apache.sysds.utils.stats.InfrastructureAnalyzer;
import org.apache.sysds.runtime.instructions.fed.FEDInstruction.FederatedOutput;

public class Explain
{
	//internal configuration parameters
	private static final boolean REPLACE_SPECIAL_CHARACTERS = true;
	private static final boolean SHOW_MEM_ABOVE_BUDGET      = true;
	private static final boolean SHOW_LITERAL_HOPS          = false;
	private static final boolean SHOW_DATA_DEPENDENCIES     = true;
	private static final boolean SHOW_DATA_FLOW_PROPERTIES  = true;

	//different explain levels
	public enum ExplainType {
		NONE, 	  // explain disabled
		HOPS,     // explain program and hops
		RUNTIME,  // explain runtime program (default)
		RECOMPILE_HOPS, // explain hops, incl recompile
		RECOMPILE_RUNTIME,  // explain runtime program, incl recompile
		CODEGEN,	// show generated code, incl runtime explanation
		CODEGEN_RECOMPILE;	// show generated code, incl runtime explanation and recompilation

		public boolean isHopsType(boolean recompile) {
			return (this==RECOMPILE_HOPS || (!recompile && this==HOPS));
		}
		public boolean isRuntimeType(boolean recompile) {
			return (this==RECOMPILE_RUNTIME || (!recompile && this==RUNTIME) || (this==CODEGEN_RECOMPILE) ||(!recompile && this==CODEGEN));
		}
		public boolean isCodegenType() {
			return (this == CODEGEN || this == CODEGEN_RECOMPILE);
		}
	}

	public static class ExplainCounts {
		public int numCPInst = 0;
		public int numJobs = 0;
		public int numReblocks = 0;
		public int numChkpts = 0;
	}

	//////////////
	// public explain interface

	public static String display(DMLProgram prog, Program rtprog, ExplainType type, ExplainCounts counts) {
		if( counts == null )
			counts = countDistributedOperations(rtprog);

		//explain plan of program (hops or runtime)
		return "# EXPLAIN ("+type.name()+"):\n"
				+ Explain.explainMemoryBudget(counts)+"\n"
				+ Explain.explainDegreeOfParallelism(counts)
				+ Explain.explain(prog, rtprog, type, counts);
	}

	public static String explainMemoryBudget() {
		return explainMemoryBudget(new ExplainCounts());
	}

	public static String explainMemoryBudget(ExplainCounts counts) {
		StringBuilder sb = new StringBuilder();
		sb.append( "# Memory Budget local/remote = " );
		sb.append( OptimizerUtils.toMB(OptimizerUtils.getLocalMemBudget()) );
		sb.append( "MB/" );

		if( OptimizerUtils.isSparkExecutionMode() ) {
			//avoid unnecessary lazy spark context creation on access to memory configurations
			if( counts.numJobs-counts.numReblocks-counts.numChkpts <= 0
				|| !SparkExecutionContext.isSparkContextCreated() ) {
				sb.append( "?MB/?MB/?MB" );
			}
			else { //default
				sb.append( OptimizerUtils.toMB(SparkExecutionContext.getDataMemoryBudget(true, false)) );
				sb.append( "MB/" );
				sb.append( OptimizerUtils.toMB(SparkExecutionContext.getDataMemoryBudget(false, false)) );
				sb.append( "MB/" );
				sb.append( OptimizerUtils.toMB(SparkExecutionContext.getBroadcastMemoryBudget()) );
				sb.append( "MB" );
			}
		}
		else {
			sb.append( "?MB/?MB" );
		}

		return sb.toString();
	}

	public static String explainDegreeOfParallelism() {
		return explainDegreeOfParallelism(new ExplainCounts());
	}

	public static String explainDegreeOfParallelism(ExplainCounts counts) {
		int lk = InfrastructureAnalyzer.getLocalParallelism();
		StringBuilder sb = new StringBuilder();
		sb.append( "# Degree of Parallelism (vcores) local/remote = " );
		sb.append( lk );
		sb.append( "/" );

		if( OptimizerUtils.isSparkExecutionMode() ) {
			if( counts.numJobs-counts.numReblocks-counts.numChkpts <= 0
				|| !SparkExecutionContext.isSparkContextCreated() ) {
				//avoid unnecessary lazy spark context creation on access to memory configurations
				sb.append( "?" );
			}
			else { //default
				sb.append( SparkExecutionContext.getDefaultParallelism(false) );
			}
		}

		return sb.toString();
	}

	public static String explain(DMLProgram prog, Program rtprog, ExplainType type) {
		return explain(prog, rtprog, type, null);
	}

	public static String explain(DMLProgram prog, Program rtprog, ExplainType type, ExplainCounts counts) {
		//dispatch to individual explain utils
		switch( type ) {
			//explain hops with stats
			case HOPS:
			case RECOMPILE_HOPS:
				return explain(prog);
			//explain runtime program
			case RUNTIME:
			case RECOMPILE_RUNTIME:
			case CODEGEN:
			case CODEGEN_RECOMPILE:
				return explain(rtprog, counts);
			case NONE:
				//do nothing
		}

		return null;
	}

	public static String explain(DMLProgram prog)
	{
		StringBuilder sb = new StringBuilder();

		//create header
		sb.append("\nPROGRAM\n");

		// Explain functions (if exists)
		if( prog.hasFunctionStatementBlocks() ) {
			sb.append("--FUNCTIONS\n");

			//show function call graph
			sb.append("----FUNCTION CALL GRAPH\n");
			sb.append("------MAIN PROGRAM\n");
			FunctionCallGraph fgraph = new FunctionCallGraph(prog);
			sb.append(explainFunctionCallGraph(fgraph, new HashSet<String>(), null, 3));

			//show individual functions
			for (String namespace : prog.getNamespaces().keySet()) {
				for (String fname : prog.getFunctionStatementBlocks(namespace).keySet()) {
					FunctionStatementBlock fsb = prog.getFunctionStatementBlock(namespace, fname);
					FunctionStatement fstmt = (FunctionStatement) fsb.getStatement(0);
					String fkey = DMLProgram.constructFunctionKey(namespace, fname);
					sb.append("----FUNCTION " + fkey + " [recompile="+fsb.isRecompileOnce()+"]\n");
					for (StatementBlock current : fstmt.getBody())
						sb.append(explainStatementBlock(current, 3));
				}
			}
		}

		// Explain main program
		sb.append("--MAIN PROGRAM\n");
		for( StatementBlock sblk : prog.getStatementBlocks() )
			sb.append(explainStatementBlock(sblk, 2));

		return sb.toString();
	}

	public static String explain( Program rtprog ) {
		return explain(rtprog, null);
	}

	public static String explain( Program rtprog, ExplainCounts counts )
	{
		//counts number of instructions
		boolean sparkExec = OptimizerUtils.isSparkExecutionMode();
		if( counts == null ) {
			counts = new ExplainCounts();
			countCompiledInstructions(rtprog, counts, true, sparkExec);
		}

		StringBuilder sb = new StringBuilder();

		//create header
		sb.append("\nPROGRAM ( size CP/"+(sparkExec?"SP":"MR")+" = ");
		sb.append(counts.numCPInst);
		sb.append("/");
		sb.append(counts.numJobs);
		sb.append(" )\n");

		//explain functions (if exists)
		Map<String, FunctionProgramBlock> funcMap = rtprog.getFunctionProgramBlocks();
		if( funcMap != null && !funcMap.isEmpty() )
		{
			sb.append("--FUNCTIONS\n");

			//show function call graph
			if( !rtprog.getProgramBlocks().isEmpty() &&
					rtprog.getProgramBlocks().get(0).getStatementBlock() != null )
			{
				sb.append("----FUNCTION CALL GRAPH\n");
				sb.append("------MAIN PROGRAM\n");
				DMLProgram prog = rtprog.getProgramBlocks().get(0).getStatementBlock().getDMLProg();
				FunctionCallGraph fgraph = new FunctionCallGraph(prog);
				sb.append(explainFunctionCallGraph(fgraph, new HashSet<String>(), null, 3));
			}

			//show individual functions
			for( Entry<String, FunctionProgramBlock> e : funcMap.entrySet() ) {
				String fkey = e.getKey();
				FunctionProgramBlock fpb = e.getValue();
				//explain optimized function
				sb.append("----FUNCTION "+fkey+" [recompile="+fpb.isRecompileOnce()+"]\n");
				for( ProgramBlock pb : fpb.getChildBlocks() )
					sb.append( explainProgramBlock(pb,3) );
				//explain unoptimized function
				if( rtprog.containsFunctionProgramBlock(fkey, false) ) {
					FunctionProgramBlock fpb2 = rtprog.getFunctionProgramBlock(fkey, false);
					sb.append("----FUNCTION "+fkey+" (unoptimized) [recompile="+fpb2.isRecompileOnce()+"]\n");
					for( ProgramBlock pb : fpb2.getChildBlocks() )
						sb.append( explainProgramBlock(pb,3) );
				}
			}
		}

		//explain main program
		sb.append("--MAIN PROGRAM\n");
		for( ProgramBlock pb : rtprog.getProgramBlocks() )
			sb.append( explainProgramBlock(pb,2) );

		return sb.toString();
	}
	
	public static String explain( ProgramBlock pb ) {
		return explainProgramBlock(pb, 0);
	}

	public static String explain( List<Instruction> inst ) {
		return explainInstructions(inst, 0);
	}

	public static String explain( List<Instruction> inst, int level ) {
		return explainInstructions(inst, level);
	}

	public static String explain( Instruction inst ) {
		return explainGenericInstruction(inst, 0);
	}

	public static String explain( StatementBlock sb ) {
		return explainStatementBlock(sb, 0);
	}

	public static String explainHops( List<Hop> hops ) {
		return explainHops(hops, 0);
	}

	public static String explainHops( List<Hop> hops, int level ) {
		StringBuilder sb = new StringBuilder();
		Hop.resetVisitStatus(hops);
		for( Hop hop : hops )
			sb.append(explainHop(hop, level));
		Hop.resetVisitStatus(hops);
		return sb.toString();
	}

	public static String explain( Hop hop ) {
		return explain(hop, 0);
	}

	public static String explain( Hop hop, int level ) {
		hop.resetVisitStatus();
		String ret = explainHop(hop, level);
		hop.resetVisitStatus();
		return ret;
	}

	public static String explainLineageItems( LineageItem[] lis ) {
		return explainLineageItems(lis, 0);
	}

	public static String explainLineageItems( LineageItem[] lis, int level ) {
		StringBuilder sb = new StringBuilder();
		LineageItem.resetVisitStatusNR(lis);
		for( LineageItem li : lis )
			sb.append(explainLineageItemNR(li, level));
		LineageItem.resetVisitStatusNR(lis);
		return sb.toString();
	}

	public static String explain( LineageItem li ) {
		li.resetVisitStatusNR();
		String s = explain(li, 0);
		//s += rExplainDedupItems(li, new ArrayList<>());
		li.resetVisitStatusNR();
		return s;
	}

	private static String explain( LineageItem li, int level ) {
		li.resetVisitStatusNR();
		String ret = explainLineageItemNR(li, level);
		li.resetVisitStatusNR();
		return ret;
	}
	
	@Deprecated
	@SuppressWarnings("unused")
	private static String rExplainDedupItems(LineageItem li, List<String> paths) {
		if (li.isVisited())
			return "";
		StringBuilder sb = new StringBuilder();
		
		if (li.getType() == LineageItem.LineageItemType.Dedup && !paths.contains(li.getData())) {
			sb.append("\n").append("dedup").append(li.getData()).append(":\n");
			sb.append(Explain.explain(li, 0));
			paths.add(li.getData());
		}
		
		if (li.getInputs() != null)
			for (LineageItem in : li.getInputs())
				sb.append(rExplainDedupItems(in, paths));
		
		li.setVisited();
		return sb.toString();
	}
	

	public static String explainCPlan( CNodeTpl cplan ) {
		StringBuilder sb = new StringBuilder();

		//create template header
		sb.append("\n----------------------------------------\n");
		sb.append("CPLAN: "+cplan.getTemplateInfo()+"\n");
		sb.append("--inputs: "+Arrays.toString(cplan.getInputNames())+"\n");
		sb.append("----------------------------------------\n");

		//explain body dag
		cplan.resetVisitStatusOutputs();
		if( cplan instanceof CNodeMultiAgg )
			for( CNode output : ((CNodeMultiAgg)cplan).getOutputs() )
				sb.append(explainCNode(output, 1));
		else
			sb.append(explainCNode(cplan.getOutput(), 1));
		cplan.resetVisitStatusOutputs();
		sb.append("----------------------------------------\n");

		return sb.toString();
	}

	public static String explain( CNode node ) {
		return explain(node, 0);
	}

	public static String explain( CNode node, int level ) {
		return explainCNode(node, level);
	}

	/**
	 * Counts the number of compiled MRJob/Spark instructions in the
	 * given runtime program.
	 *
	 * @param rtprog runtime program
	 * @return counts
	 */
	public static ExplainCounts countDistributedOperations( Program rtprog ) {
		ExplainCounts counts = new ExplainCounts();
		Explain.countCompiledInstructions(rtprog, counts, true, true);
		return counts;
	}

	public static String getIdentation( int level ) {
		return createOffset(level);
	}

	//////////////
	// internal explain HOPS

	private static String explainStatementBlock(StatementBlock sb, int level)
	{
		StringBuilder builder = new StringBuilder();
		String offset = createOffset(level);

		if (sb instanceof WhileStatementBlock) {
			WhileStatementBlock wsb = (WhileStatementBlock) sb;
			builder.append(offset);
			if( !wsb.getUpdateInPlaceVars().isEmpty() || wsb.isRecompileOnce() ) {
				builder.append("WHILE (lines "+wsb.getBeginLine()+"-"+wsb.getEndLine()+") ");
				builder.append("[in-place="+wsb.getUpdateInPlaceVars().toString()+", recompile="+wsb.isRecompileOnce()+"]\n");
			}
			else
				builder.append("WHILE (lines "+wsb.getBeginLine()+"-"+wsb.getEndLine()+")\n");
			builder.append(explainHop(wsb.getPredicateHops(), level+1));

			WhileStatement ws = (WhileStatement)sb.getStatement(0);
			for (StatementBlock current : ws.getBody())
				builder.append(explainStatementBlock(current, level+1));

		}
		else if (sb instanceof IfStatementBlock) {
			IfStatementBlock ifsb = (IfStatementBlock) sb;
			builder.append(offset);
			builder.append("IF (lines "+ifsb.getBeginLine()+"-"+ifsb.getEndLine()+")\n");
			builder.append(explainHop(ifsb.getPredicateHops(), level+1));

			IfStatement ifs = (IfStatement) sb.getStatement(0);
			for (StatementBlock current : ifs.getIfBody())
				builder.append(explainStatementBlock(current, level+1));
			if( !ifs.getElseBody().isEmpty() ) {
				builder.append(offset);
				builder.append("ELSE\n");
			}
			for (StatementBlock current : ifs.getElseBody())
				builder.append(explainStatementBlock(current, level+1));

		}
		else if (sb instanceof ForStatementBlock) {
			ForStatementBlock fsb = (ForStatementBlock) sb;
			builder.append(offset);
			if (sb instanceof ParForStatementBlock) {
				if( !fsb.getUpdateInPlaceVars().isEmpty() )
					builder.append("PARFOR (lines "+fsb.getBeginLine()+"-"+fsb.getEndLine()+") [in-place="+fsb.getUpdateInPlaceVars().toString()+"]\n");
				else
					builder.append("PARFOR (lines "+fsb.getBeginLine()+"-"+fsb.getEndLine()+")\n");
			}
			else {
				if( !fsb.getUpdateInPlaceVars().isEmpty() || fsb.isRecompileOnce() ) {
					builder.append("FOR (lines "+fsb.getBeginLine()+"-"+fsb.getEndLine()+") ");
					builder.append("[in-place="+fsb.getUpdateInPlaceVars().toString()+", recompile="+fsb.isRecompileOnce()+"]\n");
				}
				else
					builder.append("FOR (lines "+fsb.getBeginLine()+"-"+fsb.getEndLine()+")\n");
			}
			if (fsb.getFromHops() != null)
				builder.append(explainHop(fsb.getFromHops(), level+1));
			if (fsb.getToHops() != null)
				builder.append(explainHop(fsb.getToHops(), level+1));
			if (fsb.getIncrementHops() != null)
				builder.append(explainHop(fsb.getIncrementHops(), level+1));

			ForStatement fs = (ForStatement)sb.getStatement(0);
			for (StatementBlock current : fs.getBody())
				builder.append(explainStatementBlock(current, level+1));

		}
		else if (sb instanceof FunctionStatementBlock) {
			FunctionStatement fsb = (FunctionStatement) sb.getStatement(0);
			for (StatementBlock current : fsb.getBody())
				builder.append(explainStatementBlock(current, level+1));

		}
		else {
			// For generic StatementBlock
			builder.append(offset);
			builder.append("GENERIC (lines "+sb.getBeginLine()+"-"+sb.getEndLine()+") [recompile=" + sb.requiresRecompilation() + "]\n");
			ArrayList<Hop> hopsDAG = sb.getHops();
			if( hopsDAG != null && !hopsDAG.isEmpty() ) {
				Hop.resetVisitStatus(hopsDAG);
				for (Hop hop : hopsDAG)
					builder.append(explainHop(hop, level+1));
				Hop.resetVisitStatus(hopsDAG);
			}
		}

		return builder.toString();
	}

	/**
	 * Do a post-order traverse through the Hop DAG and explain each Hop
	 *
	 * @param hop high-level operator
	 * @param level offset
	 * @return string explanation of Hop DAG
	 */
	private static String explainHop(Hop hop, int level) {
		if( hop.isVisited() || (!SHOW_LITERAL_HOPS && hop instanceof LiteralOp) )
			return "";

		StringBuilder sb = new StringBuilder();
		String offset = createOffset(level);

		for( Hop input : hop.getInput() )
			sb.append(explainHop(input, level));

		//indentation
		sb.append(offset);

		//hop id
		if( SHOW_DATA_DEPENDENCIES )
			sb.append("("+hop.getHopID()+") ");

		//operation string
		sb.append(hop.getOpString());

		//input hop references
		if( SHOW_DATA_DEPENDENCIES ) {
			StringBuilder childs = new StringBuilder();
			childs.append(" (");
			boolean childAdded = false;
			for( Hop input : hop.getInput() )
				if( SHOW_LITERAL_HOPS || !(input instanceof LiteralOp) ){
					childs.append(childAdded?",":"");
					childs.append(input.getHopID());
					childAdded = true;
				}
			childs.append(")");
			if( childAdded )
				sb.append(childs.toString());
		}

		//matrix characteristics
		sb.append(" [" + hop.getDim1() + ","
				+ hop.getDim2() + ","
				+ hop.getBlocksize() + ","
				+ hop.getNnz());

		if (hop.getUpdateType().isInPlace())
			sb.append("," + hop.getUpdateType().toString().toLowerCase());

		sb.append("]");

		//memory estimates
		sb.append(" [" + showMem(hop.getInputMemEstimate(), false) + ","
				+ showMem(hop.getIntermediateMemEstimate(), false) + ","
				+ showMem(hop.getOutputMemEstimate(), false) + " -> "
				+ showMem(hop.getMemEstimate(), true) + "]");

		//data flow properties
		if( SHOW_DATA_FLOW_PROPERTIES ) {
			if( hop.requiresReblock() && hop.requiresCheckpoint() )
				sb.append(" [rblk,chkpt]");
			else if( hop.requiresReblock() )
				sb.append(" [rblk]");
			else if( hop.requiresCheckpoint() )
				sb.append(" [chkpt]");
		}

		//exec type
		if (hop.getExecType() != null)
			sb.append(", " + hop.getExecType());

		if ( hop.getFederatedOutput() != FederatedOutput.NONE )
			sb.append(" ").append(hop.getFederatedOutput()).append(" ");

		sb.append('\n');

		hop.setVisited();

		return sb.toString();
	}

	private static String explainLineageItemNR(LineageItem item, int level) {
		//NOTE: in contrast to similar non-recursive functions like resetVisitStatusNR,
		// we maintain a more complex stack to ensure DFS ordering where current nodes
		// are added after the subtree underneath is processed (backwards compatibility)
		Stack<LineageItem> stackItem = new Stack<>();
		Stack<MutableInt> stackPos = new Stack<>();
		stackItem.push(item); stackPos.push(new MutableInt(0));
		StringBuilder sb = new StringBuilder();
		while( !stackItem.empty() ) {
			LineageItem tmpItem = stackItem.peek();
			MutableInt tmpPos = stackPos.peek();
			//check ascent condition - no item processing
			if( tmpItem.isVisited() ) {
				stackItem.pop(); stackPos.pop();
			}
			//check ascent condition - append item
			else if( tmpItem.getInputs() == null 
				|| tmpItem.getOpcode().startsWith(LineageItemUtils.LPLACEHOLDER)
				// don't trace beyond if a placeholder is found
				|| tmpItem.getInputs().length <= tmpPos.intValue() ) {
				sb.append(createOffset(level));
				sb.append(tmpItem.toString());
				sb.append('\n');
				stackItem.pop(); stackPos.pop();
				tmpItem.setVisited();
			}
			//check descent condition
			else if( tmpItem.getInputs() != null ) {
				stackItem.push(tmpItem.getInputs()[tmpPos.intValue()]);
				tmpPos.increment();
				stackPos.push(new MutableInt(0));
			}
		}
		return sb.toString();
	}
	
	@Deprecated
	@SuppressWarnings("unused")
	private static String explainLineageItem(LineageItem li, int level) {
		if( li.isVisited())
			return "";

		StringBuilder sb = new StringBuilder();
		String offset = createOffset(level);

		if (li.getInputs() != null)
			for( LineageItem input : li.getInputs() )
				sb.append(explainLineageItem(input, level));

		sb.append(offset);
		sb.append(li.toString());
		sb.append('\n');

		li.setVisited();

		return sb.toString();
	}

	//////////////
	// internal explain CNODE

	private static String explainCNode(CNode cnode, int level) {
		if( cnode.isVisited() )
			return "";

		StringBuilder sb = new StringBuilder();
		String offset = createOffset(level);

		for( CNode input : cnode.getInput() )
			sb.append(explainCNode(input, level));

		//indentation
		sb.append(offset);

		//hop id
		if( SHOW_DATA_DEPENDENCIES )
			sb.append("("+cnode.getID()+") ");

		//operation string
		sb.append(cnode.toString());

		//input hop references 
		if( SHOW_DATA_DEPENDENCIES ) {
			StringBuilder childs = new StringBuilder();
			childs.append(" (");
			boolean childAdded = false;
			for( CNode input : cnode.getInput() ) {
				childs.append(childAdded?",":"");
				childs.append(input.getID());
				childAdded = true;
			}
			childs.append(")");
			if( childAdded )
				sb.append(childs.toString());
		}

		sb.append('\n');
		cnode.setVisited();

		return sb.toString();
	}

	//////////////
	// internal explain RUNTIME


	public static String explainProgramBlocks( List<ProgramBlock> pbs ) {
		StringBuilder sb = new StringBuilder();
		for(ProgramBlock pb : pbs)
			sb.append(explain(pb));
		return sb.toString();
	}
	
	private static String explainProgramBlock( ProgramBlock pb, int level )
	{
		StringBuilder sb = new StringBuilder();
		String offset = createOffset(level);

		if (pb instanceof FunctionProgramBlock ) {
			FunctionProgramBlock fpb = (FunctionProgramBlock)pb;
			for( ProgramBlock pbc : fpb.getChildBlocks() )
				sb.append( explainProgramBlock( pbc, level+1) );
		}
		else if (pb instanceof WhileProgramBlock) {
			WhileProgramBlock wpb = (WhileProgramBlock) pb;
			StatementBlock wsb = pb.getStatementBlock();
			sb.append(offset);
			if( wsb != null && (!wsb.getUpdateInPlaceVars().isEmpty() || wsb.isRecompileOnce()) ) {
				sb.append("WHILE (lines "+wpb.getBeginLine()+"-"+wpb.getEndLine()+") ");
				sb.append("[in-place="+wsb.getUpdateInPlaceVars().toString()+", recompile="+wsb.isRecompileOnce()+"]\n");
			}
			else
				sb.append("WHILE (lines "+wpb.getBeginLine()+"-"+wpb.getEndLine()+")\n");
			sb.append(explainInstructions(wpb.getPredicate(), level+1));
			for( ProgramBlock pbc : wpb.getChildBlocks() )
				sb.append( explainProgramBlock( pbc, level+1) );
			if( wpb.getExitInstruction() != null )
				sb.append(explainInstructions(wpb.getExitInstruction(), level+1));
		}
		else if (pb instanceof IfProgramBlock) {
			IfProgramBlock ipb = (IfProgramBlock) pb;
			sb.append(offset);
			sb.append("IF (lines "+ipb.getBeginLine()+"-"+ipb.getEndLine()+")\n");
			sb.append(explainInstructions(ipb.getPredicate(), level+1));
			for( ProgramBlock pbc : ipb.getChildBlocksIfBody() )
				sb.append( explainProgramBlock( pbc, level+1) );
			if( !ipb.getChildBlocksElseBody().isEmpty() ) {
				sb.append(offset);
				sb.append("ELSE\n");
				for( ProgramBlock pbc : ipb.getChildBlocksElseBody() )
					sb.append( explainProgramBlock( pbc, level+1) );
			}
			if( ipb.getExitInstruction() != null )
				sb.append(explainInstructions(ipb.getExitInstruction(), level+1));
		}
		else if (pb instanceof ForProgramBlock) { //incl parfor
			ForProgramBlock fpb = (ForProgramBlock) pb;
			StatementBlock fsb = pb.getStatementBlock();
			sb.append(offset);
			if( pb instanceof ParForProgramBlock )
				sb.append("PARFOR (lines "+fpb.getBeginLine()+"-"+fpb.getEndLine()+")\n");
			else {
				if( fsb != null && (!fsb.getUpdateInPlaceVars().isEmpty() || fsb.isRecompileOnce()) ) {
					sb.append("FOR (lines "+fpb.getBeginLine()+"-"+fpb.getEndLine()+") ");
					sb.append("[in-place="+fsb.getUpdateInPlaceVars().toString()+", recompile="+fsb.isRecompileOnce()+"]\n");
				}
				else
					sb.append("FOR (lines "+fpb.getBeginLine()+"-"+fpb.getEndLine()+")\n");
			}
			sb.append(explainInstructions(fpb.getFromInstructions(), level+1));
			sb.append(explainInstructions(fpb.getToInstructions(), level+1));
			sb.append(explainInstructions(fpb.getIncrementInstructions(), level+1));
			for( ProgramBlock pbc : fpb.getChildBlocks() )
				sb.append( explainProgramBlock( pbc, level+1) );
			if( fpb.getExitInstruction() != null )
				sb.append(explainInstructions(fpb.getExitInstruction(), level+1));
		}
		else if( pb instanceof BasicProgramBlock ) {
			BasicProgramBlock bpb = (BasicProgramBlock) pb;
			sb.append(offset);
			if( pb.getStatementBlock()!=null )
				sb.append("GENERIC (lines "+pb.getBeginLine()+"-"+pb.getEndLine()+") [recompile="+pb.getStatementBlock().requiresRecompilation()+"]\n");
			else
				sb.append("GENERIC (lines "+pb.getBeginLine()+"-"+pb.getEndLine()+") \n");
			sb.append(explainInstructions(bpb.getInstructions(), level+1));
		}

		return sb.toString();
	}

	private static String explainInstructions( List<Instruction> instSet, int level ) {
		StringBuilder sb = new StringBuilder();
		String offsetInst = createOffset(level);
		for( Instruction inst : instSet ) {
			String tmp = explainGenericInstruction(inst, level);
			sb.append( offsetInst );
			sb.append( tmp );
			sb.append( '\n' );
		}

		return sb.toString();
	}

	private static String explainInstructions( Instruction inst, int level ) {
		StringBuilder sb = new StringBuilder();
		sb.append( createOffset(level) );
		sb.append( explainGenericInstruction(inst, level) );
		sb.append( '\n' );
		return sb.toString();
	}
	
	private static String explainGenericInstruction( Instruction inst, int level )
	{
		String tmp = null;
		if ( inst instanceof SPInstruction || inst instanceof CPInstruction 
			|| inst instanceof GPUInstruction || inst instanceof FEDInstruction
			|| inst instanceof OOCInstruction)
			tmp = inst.toString();

		if( REPLACE_SPECIAL_CHARACTERS && tmp != null){
			tmp = tmp.replaceAll(Lop.OPERAND_DELIMITOR, " ");
			tmp = tmp.replaceAll(Lop.DATATYPE_PREFIX, ".");
			tmp = tmp.replaceAll(Lop.INSTRUCTION_DELIMITOR, ", ");
		}

		return tmp;
	}

	@SuppressWarnings("unused")
	private static String showMem(double mem, boolean units)
	{
		if( !SHOW_MEM_ABOVE_BUDGET && mem >= OptimizerUtils.DEFAULT_SIZE )
			return "MAX";
		return OptimizerUtils.toMB(mem) + (units?"MB":"");
	}

	public static String createOffset( int level )
	{
		StringBuilder sb = new StringBuilder();
		for( int i=0; i<level; i++ )
			sb.append("--");
		return sb.toString();
	}

	private static void countCompiledInstructions( Program rtprog, ExplainCounts counts, boolean CP, boolean SP )
	{
		//analyze DML-bodied functions
		for( FunctionProgramBlock fpb : rtprog.getFunctionProgramBlocks().values() )
			countCompiledInstructions( fpb, counts, CP, SP );

		//analyze main program
		for( ProgramBlock pb : rtprog.getProgramBlocks() )
			countCompiledInstructions( pb, counts, CP, SP );
	}

	/**
	 * Recursively counts the number of compiled MRJob instructions in the
	 * given runtime program block. 
	 *
	 * @param pb program block
	 * @param counts explain countst
	 * @param CP if true, count CP instructions
	 * @param SP if true, count Spark instructions
	 */
	private static void countCompiledInstructions(ProgramBlock pb, ExplainCounts counts, boolean CP, boolean SP)
	{
		if (pb instanceof WhileProgramBlock) {
			WhileProgramBlock tmp = (WhileProgramBlock)pb;
			countCompiledInstructions(tmp.getPredicate(), counts, CP, SP);
			for (ProgramBlock pb2 : tmp.getChildBlocks())
				countCompiledInstructions(pb2, counts, CP, SP);
		}
		else if (pb instanceof IfProgramBlock) {
			IfProgramBlock tmp = (IfProgramBlock)pb;
			countCompiledInstructions(tmp.getPredicate(), counts, CP, SP);
			for( ProgramBlock pb2 : tmp.getChildBlocksIfBody() )
				countCompiledInstructions(pb2, counts, CP, SP);
			for( ProgramBlock pb2 : tmp.getChildBlocksElseBody() )
				countCompiledInstructions(pb2, counts, CP, SP);
		}
		else if (pb instanceof ForProgramBlock) { //includes ParFORProgramBlock
			ForProgramBlock tmp = (ForProgramBlock)pb;
			countCompiledInstructions(tmp.getFromInstructions(), counts, CP, SP);
			countCompiledInstructions(tmp.getToInstructions(), counts, CP, SP);
			countCompiledInstructions(tmp.getIncrementInstructions(), counts, CP, SP);
			for( ProgramBlock pb2 : tmp.getChildBlocks() )
				countCompiledInstructions(pb2, counts, CP, SP);
			//additional parfor jobs counted during runtime
		}
		else if ( pb instanceof FunctionProgramBlock ) {
			FunctionProgramBlock fpb = (FunctionProgramBlock)pb;
			for( ProgramBlock pb2 : fpb.getChildBlocks() )
				countCompiledInstructions(pb2, counts, CP, SP);
		}
		else if( pb instanceof BasicProgramBlock ) {
			BasicProgramBlock bpb = (BasicProgramBlock) pb;
			countCompiledInstructions(bpb.getInstructions(), counts, CP, SP);
		}
	}

	/**
	 * Count the number of Hadoop instructions, CP instructions, Spark
	 * instructions, and/or Spark reblock instructions in a list of
	 * instructions.
	 *
	 * @param instSet
	 *            list of instructions
	 * @param counts
	 *            explain counts
	 * @param CP
	 *            if true, count CP instructions
	 * @param SP
	 *            if true, count Spark instructions and Spark reblock
	 *            instructions
	 */
	private static void countCompiledInstructions( List<Instruction> instSet, ExplainCounts counts, boolean CP, boolean SP )
	{
		for( Instruction inst : instSet )
		{
			if( CP && inst instanceof CPInstruction )
				counts.numCPInst++;
			else if( SP && inst instanceof SPInstruction )
				counts.numJobs++;

			//keep track of reblocks (in order to prevent unnecessary spark context creation)
			if( SP && (inst instanceof CSVReblockSPInstruction || inst instanceof ReblockSPInstruction) )
				counts.numReblocks++;
			if( SP && inst instanceof CheckpointSPInstruction )
				counts.numChkpts++;
		}
	}

	public static String explainFunctionCallGraph(FunctionCallGraph fgraph, Set<String> fstack, String fkey, int level)
	{
		StringBuilder builder = new StringBuilder();
		String offset = createOffset(level);
		Collection<String> cfkeys = fgraph.getCalledFunctions(fkey);
		if( cfkeys != null ) {
			for( String cfkey : cfkeys ) {
				if( fstack.contains(cfkey) && fgraph.isRecursiveFunction(cfkey) )
					builder.append(offset + "--" + cfkey + " (recursive)\n");
				else {
					fstack.add(cfkey);
					builder.append(offset + "--" + cfkey + "\n");
					builder.append(explainFunctionCallGraph(fgraph, fstack, cfkey, level+1));
					fstack.remove(cfkey);
				}
			}
		}

		return builder.toString();
	}
}
