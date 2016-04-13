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

package org.apache.sysml.utils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.sysml.api.DMLException;
import org.apache.sysml.hops.FunctionOp;
import org.apache.sysml.hops.Hop;
import org.apache.sysml.hops.Hop.VisitStatus;
import org.apache.sysml.hops.HopsException;
import org.apache.sysml.hops.LiteralOp;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.hops.globalopt.gdfgraph.GDFLoopNode;
import org.apache.sysml.hops.globalopt.gdfgraph.GDFNode;
import org.apache.sysml.hops.globalopt.gdfgraph.GDFNode.NodeType;
import org.apache.sysml.lops.Lop;
import org.apache.sysml.parser.DMLProgram;
import org.apache.sysml.parser.ForStatement;
import org.apache.sysml.parser.ExternalFunctionStatement;
import org.apache.sysml.parser.ForStatementBlock;
import org.apache.sysml.parser.FunctionStatement;
import org.apache.sysml.parser.FunctionStatementBlock;
import org.apache.sysml.parser.IfStatement;
import org.apache.sysml.parser.IfStatementBlock;
import org.apache.sysml.parser.ParForStatementBlock;
import org.apache.sysml.parser.WhileStatement;
import org.apache.sysml.parser.WhileStatementBlock;
import org.apache.sysml.parser.LanguageException;
import org.apache.sysml.parser.StatementBlock;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.ExternalFunctionProgramBlock;
import org.apache.sysml.runtime.controlprogram.ForProgramBlock;
import org.apache.sysml.runtime.controlprogram.FunctionProgramBlock;
import org.apache.sysml.runtime.controlprogram.IfProgramBlock;
import org.apache.sysml.runtime.controlprogram.ParForProgramBlock;
import org.apache.sysml.runtime.controlprogram.Program;
import org.apache.sysml.runtime.controlprogram.ProgramBlock;
import org.apache.sysml.runtime.controlprogram.WhileProgramBlock;
import org.apache.sysml.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysml.runtime.instructions.Instruction;
import org.apache.sysml.runtime.instructions.MRJobInstruction;
import org.apache.sysml.runtime.instructions.cp.CPInstruction;
import org.apache.sysml.runtime.instructions.flink.FLInstruction;
import org.apache.sysml.runtime.instructions.spark.CSVReblockSPInstruction;
import org.apache.sysml.runtime.instructions.spark.ReblockSPInstruction;
import org.apache.sysml.runtime.instructions.spark.SPInstruction;
import org.apache.sysml.yarn.ropt.YarnClusterAnalyzer;

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
		RECOMPILE_RUNTIME, // explain runtime program, incl recompile 
	};
	
	public static class ExplainCounts {
		public int numCPInst = 0;
		public int numJobs = 0;
		public int numReblocks = 0;
	}
	
	//////////////
	// public explain interface
	
	/**
	 * 
	 * @return
	 */
	public static String explainMemoryBudget() {
		return explainMemoryBudget(new ExplainCounts());
	}
	
	/**
	 * 
	 * @return
	 */
	public static String explainMemoryBudget(ExplainCounts counts)
	{
		StringBuilder sb = new StringBuilder();
		
		sb.append( "# Memory Budget local/remote = " );
		sb.append( OptimizerUtils.toMB(OptimizerUtils.getLocalMemBudget()) );
		sb.append( "MB/" );
		
		if( OptimizerUtils.isSparkExecutionMode() )
		{
			if( counts.numJobs-counts.numReblocks == 0 ) {
				//avoid unnecessary lazy spark context creation on access to memory configurations
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
		else
		{
			sb.append( OptimizerUtils.toMB(OptimizerUtils.getRemoteMemBudgetMap()) );
			sb.append( "MB/" );
			sb.append( OptimizerUtils.toMB(OptimizerUtils.getRemoteMemBudgetReduce()) );
			sb.append( "MB" );
		}
		
		return sb.toString();		 
	}
	
	/**
	 * 
	 * @return
	 */
	public static String explainDegreeOfParallelism()
	{
		return explainDegreeOfParallelism(new ExplainCounts());
	}
	
	/**
	 * 
	 * @return
	 */
	public static String explainDegreeOfParallelism(ExplainCounts counts)
	{
		int lk = InfrastructureAnalyzer.getLocalParallelism();
		
		StringBuilder sb = new StringBuilder();
		sb.append( "# Degree of Parallelism (vcores) local/remote = " );
		sb.append( lk );
		sb.append( "/" );
		
		if( OptimizerUtils.isSparkExecutionMode() ) //SP
		{
			if( counts.numJobs-counts.numReblocks == 0 ) {
				//avoid unnecessary lazy spark context creation on access to memory configurations
				sb.append( "?" );
			}
			else { //default
				sb.append( SparkExecutionContext.getDefaultParallelism(false) );	
			}
		}
		else //MR
		{
			int rk = InfrastructureAnalyzer.getRemoteParallelMapTasks();
			int rk2 = InfrastructureAnalyzer.getRemoteParallelReduceTasks();
			
			//correction max number of mappers/reducers on yarn clusters
			if( InfrastructureAnalyzer.isYarnEnabled() ){
				rk = (int)Math.max(rk, YarnClusterAnalyzer.getNumCores());
				rk2 = (int)Math.max(rk2, YarnClusterAnalyzer.getNumCores()/2);
			}
			
			sb.append( rk );
			sb.append( "/" );
			sb.append( rk2 );
		}
		
		return sb.toString();		 
	}
	
	/**
	 * 
	 * @param prog
	 * @param rtprog
	 * @param type
	 * @return
	 * @throws LanguageException 
	 * @throws DMLRuntimeException 
	 * @throws HopsException 
	 */
	public static String explain(DMLProgram prog, Program rtprog, ExplainType type) 
		throws HopsException, DMLRuntimeException, LanguageException
	{
		//dispatch to individual explain utils
		switch( type ) {
			//explain hops with stats
			case HOPS:     	
			case RECOMPILE_HOPS:	
				return explain(prog);
			//explain runtime program	
			case RUNTIME:  
			case RECOMPILE_RUNTIME: 
				return explain(rtprog);
			case NONE:
				//do nothing
		}
		
		return null;
	}


	/**
	 * 
	 * @param dmlp
	 * @return
	 * @throws LanguageException 
	 * @throws HopsException 
	 * @throws DMLRuntimeException 
	 */
	public static String explain(DMLProgram prog) 
		throws HopsException, DMLRuntimeException, LanguageException 
	{
		StringBuilder sb = new StringBuilder();
		
		//create header
		sb.append("\nPROGRAM\n");
						
		// Explain functions (if exists)
		boolean firstFunction = true;
		for (String namespace : prog.getNamespaces().keySet()){
			for (String fname : prog.getFunctionStatementBlocks(namespace).keySet()){
				if (firstFunction) {
					sb.append("--FUNCTIONS\n");
					firstFunction = false;
				}
				
				//show function call dag
				sb.append("----FUNCTION CALL DAG\n");
				sb.append("------MAIN PROGRAM\n");
				HashSet<String> fstack = new HashSet<String>();
				HashSet<String> lfset = new HashSet<String>();
				for( StatementBlock sblk : prog.getStatementBlocks() )
					sb.append(explainFunctionCallDag(sblk, fstack, lfset, 3));
				
				//show individual functions
				FunctionStatementBlock fsb = prog.getFunctionStatementBlock(namespace, fname);
				FunctionStatement fstmt = (FunctionStatement) fsb.getStatement(0);
				
				if (fstmt instanceof ExternalFunctionStatement)
					sb.append("----EXTERNAL FUNCTION " + namespace + "::" + fname + "\n");
				else {
					sb.append("----FUNCTION " + namespace + "::" + fname + " [recompile="+fsb.isRecompileOnce()+"]\n");
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
	

	/**
	 * 
	 * @param rtprog
	 * @return
	 * @throws HopsException 
	 */
	public static String explain( Program rtprog ) 
		throws HopsException 
	{
		//counts number of instructions
		boolean sparkExec = OptimizerUtils.isSparkExecutionMode();
		ExplainCounts counts = new ExplainCounts();
		countCompiledInstructions(rtprog, counts, !sparkExec, true, sparkExec);
	
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
			
			//show function call dag
			if( !rtprog.getProgramBlocks().isEmpty() &&
				rtprog.getProgramBlocks().get(0).getStatementBlock() != null )
			{
				sb.append("----FUNCTION CALL DAG\n");
				sb.append("------MAIN PROGRAM\n");
				DMLProgram prog = rtprog.getProgramBlocks().get(0).getStatementBlock().getDMLProg();
				HashSet<String> fstack = new HashSet<String>();
				HashSet<String> lfset = new HashSet<String>();
				for( StatementBlock sblk : prog.getStatementBlocks() )
					sb.append(explainFunctionCallDag(sblk, fstack, lfset, 3));
			}
			
			//show individual functions
			for( Entry<String, FunctionProgramBlock> e : funcMap.entrySet() )
			{
				String fkey = e.getKey();
				FunctionProgramBlock fpb = e.getValue();
				if( fpb instanceof ExternalFunctionProgramBlock )
					sb.append("----EXTERNAL FUNCTION "+fkey+"\n");
				else
				{
					sb.append("----FUNCTION "+fkey+" [recompile="+fpb.isRecompileOnce()+"]\n");
					for( ProgramBlock pb : fpb.getChildBlocks() )
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

	/**
	 * 
	 * @param pb
	 * @return
	 */
	public static String explain( ProgramBlock pb )
	{
		return explainProgramBlock(pb, 0);
	}
	
	/**
	 * 
	 * @param inst
	 * @return
	 */
	public static String explain( ArrayList<Instruction> inst )
	{
		return explainInstructions(inst, 0);
	}
	
	/**
	 * 
	 * @param inst
	 * @param level
	 * @return
	 */
	public static String explain( ArrayList<Instruction> inst, int level )
	{
		return explainInstructions(inst, level);
	}
	
	/**
	 * 
	 * @param inst
	 * @return
	 */
	public static String explain( Instruction inst )
	{
		return explainGenericInstruction(inst, 0);
	}
	
	/**
	 * 
	 * @param sb
	 * @return
	 * @throws DMLRuntimeException 
	 * @throws HopsException 
	 */
	public static String explain( StatementBlock sb ) 
		throws HopsException, DMLRuntimeException
	{
		return explainStatementBlock(sb, 0);
	}
	
	/**
	 * 
	 * @param hops
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static String explainHops( ArrayList<Hop> hops ) 
		throws DMLRuntimeException
	{
		return explainHops(hops, 0);
	}
	
	/**
	 * 
	 * @param hops
	 * @param level
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static String explainHops( ArrayList<Hop> hops, int level ) 
		throws DMLRuntimeException
	{
		StringBuilder sb = new StringBuilder();
		
		Hop.resetVisitStatus(hops);
		for( Hop hop : hops )
			sb.append(explainHop(hop, level));
		Hop.resetVisitStatus(hops);
		
		return sb.toString();		
	}
	
	/**
	 * 
	 * @param hop
	 * @return
	 * @throws DMLRuntimeException 
	 */
	public static String explain( Hop hop ) 
		throws DMLRuntimeException
	{
		return explain(hop, 0);
	}
	
	/**
	 * 
	 * @param hop
	 * @return
	 * @throws DMLRuntimeException 
	 */
	public static String explain( Hop hop, int level ) 
		throws DMLRuntimeException
	{
		hop.resetVisitStatus();
		String ret = explainHop(hop, level);
		hop.resetVisitStatus();
		
		return ret;
	}
	
	/**
	 * 
	 * @param gdfnodes
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static String explainGDFNodes( ArrayList<GDFNode> gdfnodes ) 
		throws DMLRuntimeException
	{
		return explainGDFNodes(gdfnodes, 0);
	}
	
	/**
	 * 
	 * @param gdfnodes
	 * @param level
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static String explainGDFNodes( ArrayList<GDFNode> gdfnodes, int level ) 
		throws DMLRuntimeException
	{
		StringBuilder sb = new StringBuilder();
		
		HashSet<Long> memo = new HashSet<Long>();
		for( GDFNode gnode : gdfnodes )
			sb.append(explainGDFNode(gnode, level, memo));
		
		return sb.toString();		
	}
	
	/**
	 * Counts the number of compiled MRJob/Spark instructions in the
	 * given runtime program.
	 * 
	 * @param rtprog
	 * @return
	 */
	public static ExplainCounts countDistributedOperations( Program rtprog )
	{		
		ExplainCounts counts = new ExplainCounts();
		if( OptimizerUtils.isSparkExecutionMode() ) 
			Explain.countCompiledInstructions(rtprog, counts, false, true, true);
		else
			Explain.countCompiledInstructions(rtprog, counts, true, true, false);
				
		return counts;		
	}
	
	/**
	 * 
	 * @param arg
	 * @return
	 * @throws DMLException
	 */
	public static ExplainType parseExplainType( String arg ) 
		throws DMLException
	{
		ExplainType ret = ExplainType.NONE;
		
		if( arg !=null )
		{
			if( arg.equalsIgnoreCase("hops") )
				ret = ExplainType.HOPS;
			else if( arg.equalsIgnoreCase("runtime") )
				ret = ExplainType.RUNTIME;
			else if( arg.equalsIgnoreCase("recompile_hops") )
				ret = ExplainType.RECOMPILE_HOPS;
			else if( arg.equalsIgnoreCase("recompile_runtime") )
				ret = ExplainType.RECOMPILE_RUNTIME;
			else 
				throw new DMLException("Failed to parse explain type: "+arg+" " +
						               "(valid types: hops, runtime, recompile_hops, recompile_runtime).");
		}
		
		return ret;
	}
	
	/**
	 * 
	 * @param level
	 * @return
	 */
	public static String getIdentation( int level ) {
		return createOffset(level);
	}
	
	//////////////
	// internal explain HOPS

	private static String explainStatementBlock(StatementBlock sb, int level) 
		throws HopsException, DMLRuntimeException 
	{
		StringBuilder builder = new StringBuilder();
		String offset = createOffset(level);
		
		if (sb instanceof WhileStatementBlock) {
			WhileStatementBlock wsb = (WhileStatementBlock) sb;
			builder.append(offset);
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
			if (sb instanceof ParForStatementBlock)
				builder.append("PARFOR (lines "+fsb.getBeginLine()+"-"+fsb.getEndLine()+")\n");
			else
				builder.append("FOR (lines "+fsb.getBeginLine()+"-"+fsb.getEndLine()+")\n");
			
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
			ArrayList<Hop> hopsDAG = sb.get_hops();
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
	 * Do a post-order traverse through the HopDag and explain each Hop
	 * 
	 * @param hop
	 * @param level
	 * @return
	 * @throws DMLRuntimeException
	 */
	private static String explainHop(Hop hop, int level) 
		throws DMLRuntimeException 
	{
		if(   hop.getVisited() == VisitStatus.DONE 
		   || (!SHOW_LITERAL_HOPS && hop instanceof LiteralOp) )
		{
			return "";
		}
		
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
				if( !(input instanceof LiteralOp) ){
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
				       + hop.getRowsInBlock() + "," 
		               + hop.getColsInBlock() + "," 
				       + hop.getNnz());
		
		if (hop.getUpdateInPlace())
			sb.append("," + hop.getUpdateInPlace());
		
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
		
		sb.append('\n');
		
		hop.setVisited(VisitStatus.DONE);
		
		return sb.toString();
	}

	//////////////
	// internal explain GDFNODE

	/**
	 * Do a post-order traverse through the GDFNode DAG and explain each GDFNode.
	 * Note: nodes referring to literalops are suppressed.
	 * 
	 * @param hop
	 * @param level
	 * @return
	 * @throws DMLRuntimeException
	 */
	private static String explainGDFNode(GDFNode gnode, int level, HashSet<Long> memo) 
		throws DMLRuntimeException 
	{
		//basic memoization via memo table since gnode has no visit status
		if( memo.contains(gnode.getID()) || 
			gnode.getNodeType()==NodeType.HOP_NODE && gnode.getHop() instanceof LiteralOp ) 
		{
			return "";
		}
		
		StringBuilder sb = new StringBuilder();
		String offset = createOffset(level);
		
		for( GDFNode input : gnode.getInputs() )
			sb.append(explainGDFNode(input, level, memo));
		
		//indentation
		sb.append(offset);
		
		//hop id
		String deps = null;
		if( SHOW_DATA_DEPENDENCIES ) {
			sb.append("("+gnode.getID()+") ");
		
			StringBuilder childs = new StringBuilder();
			childs.append(" (");
			boolean childAdded = false;
			for( GDFNode input : gnode.getInputs() ) {
				childs.append(childAdded?",":"");
				childs.append(input.getID());
				childAdded = true;
			}
			childs.append(")");		
			if( childAdded )
				deps = childs.toString();
		}
		
		//operation string
		if( gnode instanceof GDFLoopNode ) //LOOP NODES
		{
			GDFLoopNode lgnode = (GDFLoopNode) gnode;
			String offset2 = createOffset(level+1);
			sb.append(lgnode.explain(deps)+"\n"); //loop header
			sb.append(offset2+"PRED:\n");
			sb.append(explainGDFNode(lgnode.getLoopPredicate(),level+2, memo));
			sb.append(offset2+"BODY:\n");
			//note: memo table and already done child explain prevents redundancy
			for( Entry<String,GDFNode> root : lgnode.getLoopOutputs().entrySet() ) {
				sb.append(explainGDFNode(root.getValue(), level+2, memo));
			}
		}
		else //GENERAL CASE (BASIC/CROSSBLOCK NODES)
		{
			sb.append(gnode.explain(deps));
			sb.append('\n');
		}
		
		/*
		//matrix characteristics
		sb.append(" [" + hop.getDim1() + "," 
		               + hop.getDim2() + "," 
				       + hop.getRowsInBlock() + "," 
		               + hop.getColsInBlock() + "," 
				       + hop.getNnz() + "]");
		
		//memory estimates
		sb.append(" [" + showMem(hop.getInputMemEstimate(), false) + "," 
		               + showMem(hop.getIntermediateMemEstimate(), false) + "," 
				       + showMem(hop.getOutputMemEstimate(), false) + " -> " 
		               + showMem(hop.getMemEstimate(), true) + "]");
		
		//exec type
		if (hop.getExecType() != null)
			sb.append(", " + hop.getExecType());
		*/
		
		
		//memoization
		memo.add(gnode.getID());
		
		return sb.toString();
	}
	
	
	//////////////
	// internal explain RUNTIME

	/**
	 * 
	 * @param pb
	 * @param level
	 * @return
	 */
	private static String explainProgramBlock( ProgramBlock pb, int level ) 
	{
		StringBuilder sb = new StringBuilder();
		String offset = createOffset(level);
		
		if (pb instanceof FunctionProgramBlock )
		{
			FunctionProgramBlock fpb = (FunctionProgramBlock)pb;
			for( ProgramBlock pbc : fpb.getChildBlocks() )
				sb.append( explainProgramBlock( pbc, level+1) );
		}
		else if (pb instanceof WhileProgramBlock)
		{
			WhileProgramBlock wpb = (WhileProgramBlock) pb;
			sb.append(offset);
			sb.append("WHILE (lines "+wpb.getBeginLine()+"-"+wpb.getEndLine()+")\n");
			sb.append(explainInstructions(wpb.getPredicate(), level+1));			
			for( ProgramBlock pbc : wpb.getChildBlocks() )
				sb.append( explainProgramBlock( pbc, level+1) );
		}	
		else if (pb instanceof IfProgramBlock)
		{
			IfProgramBlock ipb = (IfProgramBlock) pb;
			sb.append(offset);
			sb.append("IF (lines "+ipb.getBeginLine()+"-"+ipb.getEndLine()+")\n");
			sb.append(explainInstructions(ipb.getPredicate(), level+1));
			for( ProgramBlock pbc : ipb.getChildBlocksIfBody() ) 
				sb.append( explainProgramBlock( pbc, level+1) );
			if( !ipb.getChildBlocksElseBody().isEmpty() )
			{	
				sb.append(offset);
				sb.append("ELSE\n");
				for( ProgramBlock pbc : ipb.getChildBlocksElseBody() ) 
					sb.append( explainProgramBlock( pbc, level+1) );
			}
		}
		else if (pb instanceof ForProgramBlock) //incl parfor
		{
			ForProgramBlock fpb = (ForProgramBlock) pb;
			sb.append(offset);
			if( pb instanceof ParForProgramBlock )
				sb.append("PARFOR (lines "+fpb.getBeginLine()+"-"+fpb.getEndLine()+")\n");
			else
				sb.append("FOR (lines "+fpb.getBeginLine()+"-"+fpb.getEndLine()+")\n");
			sb.append(explainInstructions(fpb.getFromInstructions(), level+1));
			sb.append(explainInstructions(fpb.getToInstructions(), level+1));
			sb.append(explainInstructions(fpb.getIncrementInstructions(), level+1));
			for( ProgramBlock pbc : fpb.getChildBlocks() ) 
				sb.append( explainProgramBlock( pbc, level+1) );
			
		}
		else
		{
			sb.append(offset);
			if( pb.getStatementBlock()!=null )
				sb.append("GENERIC (lines "+pb.getBeginLine()+"-"+pb.getEndLine()+") [recompile="+pb.getStatementBlock().requiresRecompilation()+"]\n");
			else
				sb.append("GENERIC (lines "+pb.getBeginLine()+"-"+pb.getEndLine()+") \n");
			sb.append(explainInstructions(pb.getInstructions(), level+1));
		}
		
		return sb.toString();
	}
	
	/**
	 * 
	 * @param instSet
	 * @param level
	 * @return
	 */
	private static String explainInstructions( ArrayList<Instruction> instSet, int level )
	{
		StringBuilder sb = new StringBuilder();
		String offsetInst = createOffset(level);
		
		for( Instruction inst : instSet )
		{
			String tmp = explainGenericInstruction(inst, level);
			
			sb.append( offsetInst );
			sb.append( tmp );
			sb.append( '\n' );
		}
		
		return sb.toString();
	}
	
	/**
	 * 
	 * @param inst
	 * @return
	 */
	private static String explainGenericInstruction( Instruction inst, int level )
	{
		String tmp = null;
		if( inst instanceof MRJobInstruction )
			tmp = explainMRJobInstruction((MRJobInstruction)inst, level+1);
		else if ( inst instanceof SPInstruction || inst instanceof CPInstruction || inst instanceof FLInstruction)
			tmp = inst.toString();
		
		if( REPLACE_SPECIAL_CHARACTERS ){
			tmp = tmp.replaceAll(Lop.OPERAND_DELIMITOR, " ");
			tmp = tmp.replaceAll(Lop.DATATYPE_PREFIX, ".");
			tmp = tmp.replaceAll(Lop.INSTRUCTION_DELIMITOR, ", ");
		}
		
		return tmp;
	}
	
	/**
	 * 
	 * @param inst
	 * @param level
	 * @return
	 */
	private static String explainMRJobInstruction( MRJobInstruction inst, int level )
	{		
		String instruction = "MR-Job[\n";
		String offset = createOffset(level+1);
		instruction += offset+"  jobtype        = " + inst.getJobType() + " \n";
		instruction += offset+"  input labels   = " + Arrays.toString(inst.getInputVars()) + " \n";
		instruction += offset+"  recReader inst = " + inst.getIv_recordReaderInstructions() + " \n";
		instruction += offset+"  rand inst      = " + inst.getIv_randInstructions() + " \n";
		instruction += offset+"  mapper inst    = " + inst.getIv_instructionsInMapper() + " \n";
		instruction += offset+"  shuffle inst   = " + inst.getIv_shuffleInstructions() + " \n";
		instruction += offset+"  agg inst       = " + inst.getIv_aggInstructions() + " \n";
		instruction += offset+"  other inst     = " + inst.getIv_otherInstructions() + " \n";
		instruction += offset+"  output labels  = " + Arrays.toString(inst.getOutputVars()) + " \n";
		instruction += offset+"  result indices = " + inst.getString(inst.getIv_resultIndices()) + " \n";
		//instruction += offset+"result dims unknown " + getString(iv_resultDimsUnknown) + " \n";
		instruction += offset+"  num reducers   = " + inst.getIv_numReducers() + " \n";
		instruction += offset+"  replication    = " + inst.getIv_replication() + " ]";
		//instruction += offset+"]\n";
		
		return instruction;
	}
	
	/**
	 * 
	 * @param mem
	 * @return
	 */
	@SuppressWarnings("unused")
	private static String showMem(double mem, boolean units) 
	{
		if( !SHOW_MEM_ABOVE_BUDGET && mem >= OptimizerUtils.DEFAULT_SIZE )
			return "MAX";
		return OptimizerUtils.toMB(mem) + (units?"MB":"");
	}
	
	/**
	 * 
	 * @param level
	 * @return
	 */
	private static String createOffset( int level )
	{
		StringBuilder sb = new StringBuilder();
		for( int i=0; i<level; i++ )
			sb.append("--");
		return sb.toString();
	}
	
	/**
	 * 
	 * @param rtprog
	 * @param counts
	 * @param MR
	 * @param CP
	 * @param SP
	 */
	private static void countCompiledInstructions( Program rtprog, ExplainCounts counts, boolean MR, boolean CP, boolean SP )
	{
		//analyze DML-bodied functions
		for( FunctionProgramBlock fpb : rtprog.getFunctionProgramBlocks().values() )
			countCompiledInstructions( fpb, counts, MR, CP, SP );
			
		//analyze main program
		for( ProgramBlock pb : rtprog.getProgramBlocks() ) 
			countCompiledInstructions( pb, counts, MR, CP, SP ); 
	}
	
	/**
	 * Recursively counts the number of compiled MRJob instructions in the
	 * given runtime program block. 
	 * 
	 * @param pb
	 * @param counts
	 * @param MR
	 * @param CP
	 * @param SP
	 */
	private static void countCompiledInstructions(ProgramBlock pb, ExplainCounts counts, boolean MR, boolean CP, boolean SP) 
	{
		if (pb instanceof WhileProgramBlock)
		{
			WhileProgramBlock tmp = (WhileProgramBlock)pb;
			countCompiledInstructions(tmp.getPredicate(), counts, MR, CP, SP);
			for (ProgramBlock pb2 : tmp.getChildBlocks())
				countCompiledInstructions(pb2, counts, MR, CP, SP);
		}
		else if (pb instanceof IfProgramBlock)
		{
			IfProgramBlock tmp = (IfProgramBlock)pb;	
			countCompiledInstructions(tmp.getPredicate(), counts, MR, CP, SP);
			for( ProgramBlock pb2 : tmp.getChildBlocksIfBody() )
				countCompiledInstructions(pb2, counts, MR, CP, SP);
			for( ProgramBlock pb2 : tmp.getChildBlocksElseBody() )
				countCompiledInstructions(pb2, counts, MR, CP, SP);
		}
		else if (pb instanceof ForProgramBlock) //includes ParFORProgramBlock
		{ 
			ForProgramBlock tmp = (ForProgramBlock)pb;	
			countCompiledInstructions(tmp.getFromInstructions(), counts, MR, CP, SP);
			countCompiledInstructions(tmp.getToInstructions(), counts, MR, CP, SP);
			countCompiledInstructions(tmp.getIncrementInstructions(), counts, MR, CP, SP);
			for( ProgramBlock pb2 : tmp.getChildBlocks() )
				countCompiledInstructions(pb2, counts, MR, CP, SP);
			//additional parfor jobs counted during runtime
		}		
		else if (  pb instanceof FunctionProgramBlock ) //includes ExternalFunctionProgramBlock and ExternalFunctionProgramBlockCP
		{
			FunctionProgramBlock fpb = (FunctionProgramBlock)pb;
			for( ProgramBlock pb2 : fpb.getChildBlocks() )
				countCompiledInstructions(pb2, counts, MR, CP, SP);
		}
		else 
		{
			countCompiledInstructions(pb.getInstructions(), counts, MR, CP, SP);
		}
	}
	
	/**
	 * 
	 * @param instSet
	 * @param counts
	 * @param MR
	 * @param CP
	 * @param SP
	 * @return
	 */
	private static int countCompiledInstructions( ArrayList<Instruction> instSet, ExplainCounts counts, boolean MR, boolean CP, boolean SP )
	{
		int ret = 0;
		
		for( Instruction inst : instSet )
		{
			if( MR && inst instanceof MRJobInstruction ) 
				counts.numJobs++;
			else if( SP && inst instanceof CPInstruction )
				counts.numCPInst++;
			else if( SP && inst instanceof SPInstruction )
				counts.numJobs++;
			
			//keep track of reblocks (in order to prevent unnecessary spark context creation)
			if( SP && (inst instanceof CSVReblockSPInstruction || inst instanceof ReblockSPInstruction) )
				counts.numReblocks++;
		}
		
		return ret;
	}
	
	/**
	 * 
	 * @param sb
	 * @param fstack
	 * @param lfset
	 * @param level
	 * @return
	 * @throws HopsException
	 */
	private static String explainFunctionCallDag(StatementBlock sb, HashSet<String> fstack, HashSet<String> lfset, int level) 
		throws HopsException 
	{
		StringBuilder builder = new StringBuilder();
		
		if (sb instanceof WhileStatementBlock) {
			WhileStatement ws = (WhileStatement)sb.getStatement(0);
			for (StatementBlock current : ws.getBody())
				builder.append(explainFunctionCallDag(current, fstack, lfset, level));
		} 
		else if (sb instanceof IfStatementBlock) {
			IfStatement ifs = (IfStatement) sb.getStatement(0);
			for (StatementBlock current : ifs.getIfBody())
				builder.append(explainFunctionCallDag(current, fstack, lfset, level));
			for (StatementBlock current : ifs.getElseBody())
				builder.append(explainFunctionCallDag(current, fstack, lfset, level));
		} 
		else if (sb instanceof ForStatementBlock) {
			ForStatement fs = (ForStatement)sb.getStatement(0);
			for (StatementBlock current : fs.getBody())
				builder.append(explainFunctionCallDag(current, fstack, lfset, level));
		} 
		else if (sb instanceof FunctionStatementBlock) {
			FunctionStatement fsb = (FunctionStatement) sb.getStatement(0);
			for (StatementBlock current : fsb.getBody())
				builder.append(explainFunctionCallDag(current, fstack, lfset, level));
		} 
		else {
			// For generic StatementBlock
			ArrayList<Hop> hopsDAG = sb.get_hops();
			if( hopsDAG != null && !hopsDAG.isEmpty() ) {
				//function ops can only occur as root nodes of the dag
				for( Hop h : hopsDAG )
					if( h instanceof FunctionOp ){
						FunctionOp fop = (FunctionOp) h;
						String fkey = DMLProgram.constructFunctionKey(fop.getFunctionNamespace(), fop.getFunctionName());
						//prevent redundant call edges
						if( !lfset.contains(fkey) )
						{
							//recursively explain function call dag
							if( !fstack.contains(fkey) ) {
								fstack.add(fkey);
								String offset = createOffset(level);
								builder.append(offset + "--" + fkey + "\n");
								FunctionStatementBlock fsb = sb.getDMLProg()
										.getFunctionStatementBlock(fop.getFunctionNamespace(), fop.getFunctionName());
								FunctionStatement fs = (FunctionStatement) fsb.getStatement(0);
								HashSet<String> lfset2 = new HashSet<String>(); 
								for( StatementBlock csb : fs.getBody() )
									builder.append(explainFunctionCallDag(csb, fstack, lfset2, level+1));
								fstack.remove(fkey);
							}
							//recursive function call
							else {
								String offset = createOffset(level);
								builder.append(offset + "-->" + fkey + " (recursive)\n");
							}
							
							//mark as visited for current function call context
							lfset.add( fkey );
						}
					}
			}
		}

		return builder.toString();
	}
}
