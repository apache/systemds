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

package org.apache.sysds.runtime.controlprogram.parfor.opt;

import java.util.ArrayList;

import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.IndexingOp;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.hops.codegen.SpoofCompiler;
import org.apache.sysds.hops.codegen.SpoofCompiler.IntegrationType;
import org.apache.sysds.hops.recompile.Recompiler;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.lops.LopProperties;
import org.apache.sysds.parser.DMLProgram;
import org.apache.sysds.parser.DMLTranslator;
import org.apache.sysds.parser.ForStatement;
import org.apache.sysds.parser.ForStatementBlock;
import org.apache.sysds.parser.IfStatement;
import org.apache.sysds.parser.IfStatementBlock;
import org.apache.sysds.parser.StatementBlock;
import org.apache.sysds.parser.WhileStatement;
import org.apache.sysds.parser.WhileStatementBlock;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.BasicProgramBlock;
import org.apache.sysds.runtime.controlprogram.ForProgramBlock;
import org.apache.sysds.runtime.controlprogram.IfProgramBlock;
import org.apache.sysds.runtime.controlprogram.LocalVariableMap;
import org.apache.sysds.runtime.controlprogram.Program;
import org.apache.sysds.runtime.controlprogram.ProgramBlock;
import org.apache.sysds.runtime.controlprogram.WhileProgramBlock;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.instructions.cp.BinaryCPInstruction;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.instructions.cp.FunctionCallCPInstruction;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;

public class ProgramRecompiler 
{
	public static ArrayList<ProgramBlock> generatePartitialRuntimeProgram(Program rtprog, ArrayList<StatementBlock> sbs) 
	{
		ArrayList<ProgramBlock> ret = new ArrayList<>();
		DMLConfig config = ConfigurationManager.getDMLConfig();
		
		//construct lops from hops if not existing
		DMLTranslator dmlt = new DMLTranslator(sbs.get(0).getDMLProg());
		for( StatementBlock sb : sbs ) {
			dmlt.constructLops(sb);
		}
		
		//construct runtime program from lops
		for( StatementBlock sb : sbs ) {
			ret.add(dmlt.createRuntimeProgramBlock(rtprog, sb, config));
		}
		
		//enhance runtime program by automatic operator fusion
		if( ConfigurationManager.isCodegenEnabled() 
			&& SpoofCompiler.INTEGRATION==IntegrationType.RUNTIME ) {
			for( ProgramBlock pb : ret )
				dmlt.codgenHopsDAG(pb);
		}
		
		return ret;
	}
	
	
	/**
	 * NOTE: if force is set, we set and recompile the respective indexing hops;
	 * otherwise, we release the forced exec type and recompile again. Hence, 
	 * any changes can be exactly reverted with the same access behavior.
	 * 
	 * @param sb statement block
	 * @param pb program block
	 * @param var variable
	 * @param ec execution context
	 * @param force if true, set and recompile the respective indexing hops
	 */
	public static void rFindAndRecompileIndexingHOP( StatementBlock sb, ProgramBlock pb, String var, ExecutionContext ec, boolean force )
	{
		if( pb instanceof IfProgramBlock && sb instanceof IfStatementBlock ) {
			IfProgramBlock ipb = (IfProgramBlock) pb;
			IfStatementBlock isb = (IfStatementBlock) sb;
			IfStatement is = (IfStatement) sb.getStatement(0);
			//process if condition
			if( isb.getPredicateHops()!=null )
				ipb.setPredicate( rFindAndRecompileIndexingHOP(isb.getPredicateHops(),ipb.getPredicate(),var,ec,force) );
			//process if branch
			int len = is.getIfBody().size(); 
			for( int i=0; i<ipb.getChildBlocksIfBody().size() && i<len; i++ ) {
				ProgramBlock lpb = ipb.getChildBlocksIfBody().get(i);
				StatementBlock lsb = is.getIfBody().get(i);
				rFindAndRecompileIndexingHOP(lsb,lpb,var,ec,force);
			}
			//process else branch
			if( ipb.getChildBlocksElseBody() != null ) {
				int len2 = is.getElseBody().size();
				for( int i=0; i<ipb.getChildBlocksElseBody().size() && i<len2; i++ ) {
					ProgramBlock lpb = ipb.getChildBlocksElseBody().get(i);
					StatementBlock lsb = is.getElseBody().get(i);
					rFindAndRecompileIndexingHOP(lsb,lpb,var,ec,force);
				}
			}
		}
		else if( pb instanceof WhileProgramBlock && sb instanceof WhileStatementBlock ) {
			WhileProgramBlock wpb = (WhileProgramBlock) pb;
			WhileStatementBlock wsb = (WhileStatementBlock) sb;
			WhileStatement ws = (WhileStatement) sb.getStatement(0);
			//process while condition
			if( wsb.getPredicateHops()!=null )
				wpb.setPredicate( rFindAndRecompileIndexingHOP(wsb.getPredicateHops(),wpb.getPredicate(),var,ec,force) );
			//process body
			int len = ws.getBody().size(); //robustness for potentially added problem blocks
			for( int i=0; i<wpb.getChildBlocks().size() && i<len; i++ )
			{
				ProgramBlock lpb = wpb.getChildBlocks().get(i);
				StatementBlock lsb = ws.getBody().get(i);
				rFindAndRecompileIndexingHOP(lsb,lpb,var,ec, force);
			}
		}
		else if( pb instanceof ForProgramBlock && sb instanceof ForStatementBlock ) { //for or parfor
			ForProgramBlock fpb = (ForProgramBlock) pb;
			ForStatementBlock fsb = (ForStatementBlock)sb;
			ForStatement fs = (ForStatement) fsb.getStatement(0);
			if( fsb.getFromHops()!=null )
				fpb.setFromInstructions( rFindAndRecompileIndexingHOP(fsb.getFromHops(),fpb.getFromInstructions(),var,ec,force) );
			if( fsb.getToHops()!=null )
				fpb.setToInstructions( rFindAndRecompileIndexingHOP(fsb.getToHops(),fpb.getToInstructions(),var,ec,force) );
			if( fsb.getIncrementHops()!=null )
				fpb.setIncrementInstructions( rFindAndRecompileIndexingHOP(fsb.getIncrementHops(),fpb.getIncrementInstructions(),var,ec,force) );
			//process body
			int len = fs.getBody().size(); //robustness for potentially added problem blocks
			for( int i=0; i<fpb.getChildBlocks().size() && i<len; i++ ) {
				ProgramBlock lpb = fpb.getChildBlocks().get(i);
				StatementBlock lsb = fs.getBody().get(i);
				rFindAndRecompileIndexingHOP(lsb,lpb,var,ec, force);
			}
		}
		else if( pb instanceof BasicProgramBlock ) {
			BasicProgramBlock bpb = (BasicProgramBlock) pb;
			try
			{
				//process actual hops
				boolean ret = false;
				Hop.resetVisitStatus(sb.getHops());
				if( force ) {
					//set forced execution type
					for( Hop h : sb.getHops() )
						ret |= rFindAndSetCPIndexingHOP(h, var);
				}
				else {
					//release forced execution type
					for( Hop h : sb.getHops() )
						ret |= rFindAndReleaseIndexingHOP(h, var);
				}
				
				//recompilation on-demand
				if( ret ) {
					//construct new instructions
					ArrayList<Instruction> newInst = Recompiler.recompileHopsDag(
						sb, sb.getHops(), ec.getVariables(), null, true, false, 0);
					bpb.setInstructions( newInst ); 
				}
			}
			catch(Exception ex)
			{
				throw new DMLRuntimeException(ex);
			}
		}
	}

	public static LocalVariableMap getReusableScalarVariables( DMLProgram prog, StatementBlock parforSB, LocalVariableMap vars ) {
		LocalVariableMap constVars = new LocalVariableMap(); 
		
		for( String varname : vars.keySet() )
		{
			Data dat = vars.get(varname);
			if( dat instanceof ScalarObject //scalar
				&& isApplicableForReuseVariable(prog, parforSB, varname) ) //constant
			{
				constVars.put(varname, dat);
			}
		}
		
		return constVars;
	}
	
	public static void replaceConstantScalarVariables( StatementBlock sb, LocalVariableMap vars )
	{
		if( sb instanceof IfStatementBlock )
		{
			IfStatementBlock isb = (IfStatementBlock) sb;
			IfStatement is = (IfStatement) sb.getStatement(0);
			replacePredicateLiterals(isb.getPredicateHops(), vars);
			for( StatementBlock lsb : is.getIfBody() )
				replaceConstantScalarVariables(lsb, vars);
			for( StatementBlock lsb : is.getElseBody() )
				replaceConstantScalarVariables(lsb, vars);
		}
		else if( sb instanceof WhileStatementBlock )
		{
			WhileStatementBlock wsb = (WhileStatementBlock) sb;
			WhileStatement ws = (WhileStatement) sb.getStatement(0);
			replacePredicateLiterals(wsb.getPredicateHops(), vars);
			for( StatementBlock lsb : ws.getBody() )
				replaceConstantScalarVariables(lsb, vars);		
		}
		else if( sb instanceof ForStatementBlock ) //for or parfor
		{
			ForStatementBlock fsb = (ForStatementBlock)sb;
			ForStatement fs = (ForStatement) fsb.getStatement(0);
			replacePredicateLiterals(fsb.getFromHops(), vars);
			replacePredicateLiterals(fsb.getToHops(), vars);
			replacePredicateLiterals(fsb.getIncrementHops(), vars);
			for( StatementBlock lsb : fs.getBody() )
				replaceConstantScalarVariables(lsb, vars);
		}
		else //last level block
		{
			ArrayList<Hop> hops = sb.getHops();
			if( hops != null ) 
			{	
				//replace constant literals
				Hop.resetVisitStatus(hops);
				for( Hop hopRoot : hops )
					Recompiler.rReplaceLiterals( hopRoot, vars, true );
			}	
		}
	}

	private static void replacePredicateLiterals( Hop pred, LocalVariableMap vars ) {
		if( pred != null ){
			pred.resetVisitStatus();
			Recompiler.rReplaceLiterals(pred, vars, true);
		}
	}
	
	/**
	 * This function determines if an parfor input variable is guaranteed to be read-only
	 * across multiple invocations of parfor optimization (e.g., in a surrounding while loop).
	 * In case of invariant variables we can reuse partitioned matrices and propagate constants
	 * for better size estimation.
	 * 
	 * @param prog dml program
	 * @param parforSB parfor statement block
	 * @param var variable
	 * @return true if can reuse variable
	 */
	public static boolean isApplicableForReuseVariable( DMLProgram prog, StatementBlock parforSB, String var ) {
		boolean ret = false;
		for( StatementBlock sb : prog.getStatementBlocks() )
			ret |= isApplicableForReuseVariable(sb, parforSB, var);
		return  ret;
	}

	private static boolean isApplicableForReuseVariable( StatementBlock sb, StatementBlock parforSB, String var ) {
		boolean ret = false;
		
		if( sb instanceof IfStatementBlock ) {
			IfStatement is = (IfStatement) sb.getStatement(0);
			for( StatementBlock lsb : is.getIfBody() )
				ret |= isApplicableForReuseVariable(lsb, parforSB, var);
			for( StatementBlock lsb : is.getElseBody() )
				ret |= isApplicableForReuseVariable(lsb, parforSB, var);
		}
		else if( sb instanceof WhileStatementBlock ) {
			WhileStatement ws = (WhileStatement) sb.getStatement(0);
			for( StatementBlock lsb : ws.getBody() )
				ret |= isApplicableForReuseVariable(lsb, parforSB, var);
		}
		else if( sb instanceof ForStatementBlock ) { //for or parfor
			ForStatementBlock fsb = (ForStatementBlock)sb;
			ForStatement fs = (ForStatement) fsb.getStatement(0);
			if( fsb == parforSB ) {
				//found parfor statement 
				ret = true;
			}
			else {
				for( StatementBlock lsb : fs.getBody() )
					ret |= isApplicableForReuseVariable(lsb, parforSB, var);
			}
		}
		return  ret && !sb.variablesUpdated().containsVariable(var);
	}

	public static boolean containsAtLeastOneFunction( ProgramBlock pb )
	{
		if( pb instanceof IfProgramBlock ) {
			IfProgramBlock ipb = (IfProgramBlock) pb;
			for( ProgramBlock lpb : ipb.getChildBlocksIfBody() )
				if( containsAtLeastOneFunction(lpb) )
					return true;
			for( ProgramBlock lpb : ipb.getChildBlocksElseBody() )
				if( containsAtLeastOneFunction(lpb) )
					return true;
		}
		else if( pb instanceof WhileProgramBlock ) {
			WhileProgramBlock wpb = (WhileProgramBlock) pb;
			for( ProgramBlock lpb : wpb.getChildBlocks() )
				if( containsAtLeastOneFunction(lpb) )
					return true;
		}
		else if( pb instanceof ForProgramBlock ) { //incl parfor
			ForProgramBlock fpb = (ForProgramBlock) pb;
			for( ProgramBlock lpb : fpb.getChildBlocks() )
				if( containsAtLeastOneFunction(lpb) )
					return true;
		}
		else if( pb instanceof BasicProgramBlock ) {
			BasicProgramBlock bpb = (BasicProgramBlock) pb;
			if( bpb.getInstructions() != null )
				for( Instruction inst : bpb.getInstructions() )
					if( inst instanceof FunctionCallCPInstruction )
						return true;
		}
		
		return false;
	}

	private static ArrayList<Instruction> rFindAndRecompileIndexingHOP( Hop hop, ArrayList<Instruction> in, String var, ExecutionContext ec, boolean force ) 
	{
		ArrayList<Instruction> tmp = in;
		
		try
		{
			boolean ret = false;
			hop.resetVisitStatus();
			
			if( force ) //set forced execution type
				ret = rFindAndSetCPIndexingHOP(hop, var);
			else //release forced execution type
				ret = rFindAndReleaseIndexingHOP(hop, var);
			
			//recompilation on-demand
			if( ret )
			{
				//construct new instructions
				tmp = Recompiler.recompileHopsDag(
					hop, ec.getVariables(), null, true, false, 0);
			}
		}
		catch(Exception ex)
		{
			throw new DMLRuntimeException(ex);
		}
		
		return tmp;
	}

	private static boolean rFindAndSetCPIndexingHOP(Hop hop, String var) 
	{
		boolean ret = false;
		
		if( hop.isVisited() )
			return ret;
		
		ArrayList<Hop> in = hop.getInput();
		
		if( hop instanceof IndexingOp )
		{
			String inMatrix = hop.getInput().get(0).getName();
			if( inMatrix.equals(var) )
			{
				//NOTE: mem estimate of RIX, set to output size by parfor optmizer
				//(rowblock/colblock only applied if in total less than two blocks,
				// hence always mem_est<mem_budget)
				if( hop.getMemEstimate() < OptimizerUtils.getLocalMemBudget() )
					hop.setForcedExecType( LopProperties.ExecType.CP );
				else
					hop.setForcedExecType( LopProperties.ExecType.CP_FILE );
				
				ret = true;
			}
		}
		
		//recursive search
		if( in != null )
			for( Hop hin : in )
				ret |= rFindAndSetCPIndexingHOP(hin,var);
		
		hop.setVisited();
		
		return ret;
	}
	
	private static boolean rFindAndReleaseIndexingHOP(Hop hop, String var) 
	{
		boolean ret = false;
		
		if( hop.isVisited() )
			return ret;
		
		ArrayList<Hop> in = hop.getInput();
		
		if( hop instanceof IndexingOp )
		{
			String inMatrix = hop.getInput().get(0).getName();
			if( inMatrix.equals(var) )
			{
				hop.setForcedExecType(null);
				hop.clearMemEstimate();
				ret = true;
			}
		}
		
		//recursive search
		if( in != null )
			for( Hop hin : in )
				ret |= rFindAndReleaseIndexingHOP(hin,var);
		
		hop.setVisited();
		
		return ret;
	}
	

	///////
	// additional general-purpose functionalities

	protected static ArrayList<Instruction> createNestedParallelismToInstructionSet(String iterVar, String offset) {
		//create instruction string
		StringBuilder sb = new StringBuilder("CP"+Lop.OPERAND_DELIMITOR+"+"+Lop.OPERAND_DELIMITOR);
		sb.append(iterVar);
		sb.append(Lop.DATATYPE_PREFIX+"SCALAR"+Lop.VALUETYPE_PREFIX+"INT"+Lop.OPERAND_DELIMITOR);
		sb.append(offset);
		sb.append(Lop.DATATYPE_PREFIX+"SCALAR"+Lop.VALUETYPE_PREFIX+"INT"+Lop.OPERAND_DELIMITOR);
		sb.append(iterVar);
		sb.append(Lop.DATATYPE_PREFIX+"SCALAR"+Lop.VALUETYPE_PREFIX+"INT");
		String str = sb.toString(); 
		
		//create instruction set
		ArrayList<Instruction> tmp = new ArrayList<>();
		Instruction inst = BinaryCPInstruction.parseInstruction(str);
		tmp.add(inst);
		
		return tmp;
	}
}
