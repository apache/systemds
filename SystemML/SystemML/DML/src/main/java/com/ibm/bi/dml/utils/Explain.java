/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.utils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map.Entry;

import com.ibm.bi.dml.api.DMLException;
import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.hops.Hop.VISIT_STATUS;
import com.ibm.bi.dml.hops.HopsException;
import com.ibm.bi.dml.hops.LiteralOp;
import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.parser.DMLProgram;
import com.ibm.bi.dml.parser.ForStatement;
import com.ibm.bi.dml.parser.ExternalFunctionStatement;
import com.ibm.bi.dml.parser.ForStatementBlock;
import com.ibm.bi.dml.parser.FunctionStatement;
import com.ibm.bi.dml.parser.FunctionStatementBlock;
import com.ibm.bi.dml.parser.IfStatement;
import com.ibm.bi.dml.parser.IfStatementBlock;
import com.ibm.bi.dml.parser.ParForStatementBlock;
import com.ibm.bi.dml.parser.WhileStatement;
import com.ibm.bi.dml.parser.WhileStatementBlock;
import com.ibm.bi.dml.parser.LanguageException;
import com.ibm.bi.dml.parser.StatementBlock;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.controlprogram.CVProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.ExternalFunctionProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.ForProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.FunctionProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.IfProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.Program;
import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.WhileProgramBlock;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.MRJobInstruction;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.CPInstruction;

public class Explain 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	//internal parameters
	private static final boolean REPLACE_SPECIAL_CHARACTERS = true;	
	private static final boolean SHOW_MEM_ABOVE_BUDGET = true;
	private static final boolean SHOW_LITERAL_HOPS = false;
	

	//different explain levels
	public enum ExplainType { 
		NONE, 	  // explain disabled
		HOPS,     // explain program and hops
		RUNTIME,  // explain runtime program (default)
		RECOMPILE, // explain runtime program, incl recompile
	};
	
	//////////////
	// public explain interface
	
	/**
	 * 
	 * @return
	 */
	public static String explainMemoryBudget()
	{
		StringBuffer sb = new StringBuffer();
		sb.append( "# Memory Budget local/remote = " );
		sb.append( OptimizerUtils.toMB(OptimizerUtils.getLocalMemBudget()) );
		sb.append( "MB/" );
		sb.append( OptimizerUtils.toMB(OptimizerUtils.getRemoteMemBudgetMap()) );
		sb.append( "MB/" );
		sb.append( OptimizerUtils.toMB(OptimizerUtils.getRemoteMemBudgetReduce()) );
		sb.append( "MB" );
		
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
				return explain(prog);
			//explain runtime program	
			case RUNTIME:  
			case RECOMPILE: 
				return explain(rtprog);
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
				
				FunctionStatementBlock fsb = prog.getFunctionStatementBlock(namespace, fname);
				FunctionStatement fstmt = (FunctionStatement) fsb.getStatement(0);
				
				if (fstmt instanceof ExternalFunctionStatement)
					sb.append("----EXTERNAL FUNCTION " + namespace + "::" + fname + "\n");
				else {
					sb.append("----FUNCTION " + namespace + "::" + fname + "\n");
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
	 */
	public static String explain( Program rtprog ) 
	{
		StringBuilder sb = new StringBuilder();		
	
		//create header
		sb.append("\nPROGRAM ( size CP/MR = ");
		sb.append(countCompiledInstructions(rtprog, false, true));
		sb.append("/");
		sb.append(countCompiledInstructions(rtprog, true, false));
		sb.append(" )\n");
		
		//explain functions (if exists)
		HashMap<String, FunctionProgramBlock> funcMap = rtprog.getFunctionProgramBlocks();
		if( funcMap != null && funcMap.size()>0 )
		{
			sb.append("--FUNCTIONS\n");
			for( Entry<String, FunctionProgramBlock> e : funcMap.entrySet() )
			{
				String fkey = e.getKey();
				FunctionProgramBlock fpb = e.getValue();
				if( fpb instanceof ExternalFunctionProgramBlock )
					sb.append("----EXTERNAL FUNCTION "+fkey+"\n");
				else
				{
					sb.append("----FUNCTION "+fkey+"\n");
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
		StringBuilder sb = new StringBuilder();
		
		Hop.resetVisitStatus(hops);
		for( Hop hop : hops )
			sb.append(explainHop(hop, 0));
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
		hop.resetVisitStatus();
		String ret = explainHop(hop, 0);
		hop.resetVisitStatus();
		
		return ret;
	}
	
	/**
	 * Counts the number of compiled MRJob instructions in the
	 * given runtime program.
	 * 
	 * @param rtprog
	 * @return
	 */
	public static int countCompiledMRJobs( Program rtprog )
	{
		return countCompiledInstructions(rtprog, true, false);
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
			else if( arg.equalsIgnoreCase("recompile") )
				ret = ExplainType.RECOMPILE;
			else 
				throw new DMLException("Failed to parse explain type: "+arg+" " +
						               "(valid types: hops, runtime, recompile).");
		}
		
		return ret;
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
			builder.append("WHILE\n");
			builder.append(explainHop(wsb.getPredicateHops(), level+1));
			
			WhileStatement ws = (WhileStatement)sb.getStatement(0);
			for (StatementBlock current : ws.getBody())
				builder.append(explainStatementBlock(current, level+1));
			
		} 
		else if (sb instanceof IfStatementBlock) {
			IfStatementBlock ifsb = (IfStatementBlock) sb;
			builder.append(offset);
			builder.append("IF\n");
			builder.append(explainHop(ifsb.getPredicateHops(), level+1));
			
			IfStatement ifs = (IfStatement) sb.getStatement(0);
			for (StatementBlock current : ifs.getIfBody())
				builder.append(explainStatementBlock(current, level+1));
			if (ifs.getElseBody().size() > 0) {
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
				builder.append("PARFOR\n");
			else
				builder.append("FOR\n");
			
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
			builder.append("GENERIC [recompile=" + sb.requiresRecompilation() + "]\n");
			ArrayList<Hop> hopsDAG = sb.get_hops();
			if (hopsDAG != null && hopsDAG.size() > 0) {
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
		if(   hop.get_visited() == VISIT_STATUS.DONE 
		   || (!SHOW_LITERAL_HOPS && hop instanceof LiteralOp) )
		{
			return "";
		}
		
		StringBuilder sb = new StringBuilder();
		String offset = createOffset(level);
		
		for( Hop input : hop.getInput() )
			sb.append(explainHop(input, level));
		
		sb.append(offset);
		sb.append(hop.getOpString());
		
		//matrix characteristics
		sb.append(" [" + hop.get_dim1() + "," 
		               + hop.get_dim2() + "," 
				       + hop.get_rows_in_block() + "," 
		               + hop.get_cols_in_block() + "," 
				       + hop.getNnz() + "]");
		
		//memory estimates
		sb.append(" [" + showMem(hop.getInputMemEstimate(), false) + "," 
		               + showMem(hop.getIntermediateMemEstimate(), false) + "," 
				       + showMem(hop.getOutputMemEstimate(), false) + " -> " 
		               + showMem(hop.getMemEstimate(), true) + "]");
		
		//exec type
		if (hop.getExecType() != null)
			sb.append(", " + hop.getExecType());
		
		sb.append('\n');
		
		hop.set_visited(VISIT_STATUS.DONE);
		
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
			sb.append("WHILE\n");
			sb.append(explainInstructions(wpb.getPredicate(), level+1));			
			for( ProgramBlock pbc : wpb.getChildBlocks() )
				sb.append( explainProgramBlock( pbc, level+1) );
		}	
		else if (pb instanceof IfProgramBlock)
		{
			IfProgramBlock ipb = (IfProgramBlock) pb;
			sb.append(offset);
			sb.append("IF\n");
			sb.append(explainInstructions(ipb.getPredicate(), level+1));
			for( ProgramBlock pbc : ipb.getChildBlocksIfBody() ) 
				sb.append( explainProgramBlock( pbc, level+1) );
			if( ipb.getChildBlocksElseBody().size()>0 )
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
				sb.append("PARFOR\n");
			else
				sb.append("FOR\n");
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
				sb.append("GENERIC [recompile="+pb.getStatementBlock().requiresRecompilation()+"]\n");
			else
				sb.append("GENERIC\n");
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
		else
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
	 * @param MR
	 * @param CP
	 * @return
	 */
	private static int countCompiledInstructions( Program rtprog, boolean MR, boolean CP )
	{
		int ret = 0;
		
		//analyze DML-bodied functions
		for( FunctionProgramBlock fpb : rtprog.getFunctionProgramBlocks().values() )
			ret += countCompiledInstructions( fpb, MR, CP );
			
		//analyze main program
		for( ProgramBlock pb : rtprog.getProgramBlocks() ) 
			ret += countCompiledInstructions( pb, MR, CP ); 
		
		return ret;
	}
	
	/**
	 * Recursively counts the number of compiled MRJob instructions in the
	 * given runtime program block. 
	 * 
	 * @param pb
	 * @return
	 */
	private static int countCompiledInstructions(ProgramBlock pb, boolean MR, boolean CP) 
	{
		int ret = 0;

		if (pb instanceof WhileProgramBlock)
		{
			WhileProgramBlock tmp = (WhileProgramBlock)pb;
			ret += countCompiledInstructions(tmp.getPredicate(), MR, CP);
			for (ProgramBlock pb2 : tmp.getChildBlocks())
				ret += countCompiledInstructions(pb2,MR,CP);
		}
		else if (pb instanceof IfProgramBlock)
		{
			IfProgramBlock tmp = (IfProgramBlock)pb;	
			ret += countCompiledInstructions(tmp.getPredicate(), MR, CP);
			for( ProgramBlock pb2 : tmp.getChildBlocksIfBody() )
				ret += countCompiledInstructions(pb2,MR,CP);
			for( ProgramBlock pb2 : tmp.getChildBlocksElseBody() )
				ret += countCompiledInstructions(pb2,MR,CP);
		}
		else if (pb instanceof ForProgramBlock) //includes ParFORProgramBlock
		{ 
			ForProgramBlock tmp = (ForProgramBlock)pb;	
			ret += countCompiledInstructions(tmp.getFromInstructions(), MR, CP);
			ret += countCompiledInstructions(tmp.getToInstructions(), MR, CP);
			ret += countCompiledInstructions(tmp.getIncrementInstructions(), MR, CP);
			for( ProgramBlock pb2 : tmp.getChildBlocks() )
				ret += countCompiledInstructions(pb2,MR,CP);
			//additional parfor jobs counted during runtime
		}		
		else if (  pb instanceof FunctionProgramBlock //includes ExternalFunctionProgramBlock and ExternalFunctionProgramBlockCP
			    || pb instanceof CVProgramBlock
				//|| pb instanceof ELProgramBlock
				//|| pb instanceof ELUseProgramBlock
				)
		{
			FunctionProgramBlock fpb = (FunctionProgramBlock)pb;
			for( ProgramBlock pb2 : fpb.getChildBlocks() )
				ret += countCompiledInstructions(pb2,MR,CP);
		}
		else 
		{
			ret += countCompiledInstructions(pb.getInstructions(), MR, CP);
		}
		
		return ret;
	}
	
	/**
	 * 
	 * @param instSet
	 * @param MR
	 * @param CP
	 * @return
	 */
	private static int countCompiledInstructions( ArrayList<Instruction> instSet, boolean MR, boolean CP )
	{
		int ret = 0;
		
		for( Instruction inst : instSet )
		{
			if( MR && inst instanceof MRJobInstruction ) 
				ret++;
			if( CP && inst instanceof CPInstruction )
				ret++;
		}
		
		return ret;
	}
	
}
