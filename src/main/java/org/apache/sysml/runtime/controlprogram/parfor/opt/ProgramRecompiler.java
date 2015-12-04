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

package org.apache.sysml.runtime.controlprogram.parfor.opt;

import java.io.IOException;
import java.util.ArrayList;

import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.conf.DMLConfig;
import org.apache.sysml.hops.Hop;
import org.apache.sysml.hops.HopsException;
import org.apache.sysml.hops.IndexingOp;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.hops.Hop.VisitStatus;
import org.apache.sysml.hops.recompile.Recompiler;
import org.apache.sysml.lops.LopProperties;
import org.apache.sysml.lops.Lop;
import org.apache.sysml.lops.LopsException;
import org.apache.sysml.lops.compile.Dag;
import org.apache.sysml.parser.DMLProgram;
import org.apache.sysml.parser.DMLTranslator;
import org.apache.sysml.parser.ForStatement;
import org.apache.sysml.parser.ForStatementBlock;
import org.apache.sysml.parser.IfStatement;
import org.apache.sysml.parser.IfStatementBlock;
import org.apache.sysml.parser.StatementBlock;
import org.apache.sysml.parser.WhileStatement;
import org.apache.sysml.parser.WhileStatementBlock;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.DMLUnsupportedOperationException;
import org.apache.sysml.runtime.controlprogram.ForProgramBlock;
import org.apache.sysml.runtime.controlprogram.IfProgramBlock;
import org.apache.sysml.runtime.controlprogram.LocalVariableMap;
import org.apache.sysml.runtime.controlprogram.ParForProgramBlock;
import org.apache.sysml.runtime.controlprogram.Program;
import org.apache.sysml.runtime.controlprogram.ProgramBlock;
import org.apache.sysml.runtime.controlprogram.WhileProgramBlock;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.parfor.ProgramConverter;
import org.apache.sysml.runtime.controlprogram.parfor.opt.OptNode.NodeType;
import org.apache.sysml.runtime.instructions.Instruction;
import org.apache.sysml.runtime.instructions.cp.ArithmeticBinaryCPInstruction;
import org.apache.sysml.runtime.instructions.cp.Data;
import org.apache.sysml.runtime.instructions.cp.FunctionCallCPInstruction;
import org.apache.sysml.runtime.instructions.cp.ScalarObject;

/**
 * 
 */
public class ProgramRecompiler 
{
	
	/**
	 * 
	 * @param rtprog
	 * @param sbs
	 * @return
	 * @throws IOException 
	 * @throws DMLUnsupportedOperationException 
	 * @throws DMLRuntimeException 
	 * @throws LopsException 
	 * @throws HopsException 
	 */
	public static ArrayList<ProgramBlock> generatePartitialRuntimeProgram(Program rtprog, ArrayList<StatementBlock> sbs) 
		throws LopsException, DMLRuntimeException, DMLUnsupportedOperationException, IOException, HopsException
	{
		ArrayList<ProgramBlock> ret = new ArrayList<ProgramBlock>();
		DMLConfig config = ConfigurationManager.getConfig();
		
		//construct lops from hops if not existing
		DMLTranslator dmlt = new DMLTranslator(sbs.get(0).getDMLProg());
		for( StatementBlock sb : sbs ) {
			dmlt.constructLops(sb);
		}
		
		//construct runtime program from lops
		for( StatementBlock sb : sbs ) {
			DMLProgram prog = sb.getDMLProg();
			ret.add( prog.createRuntimeProgramBlock(rtprog, sb, config) );
		}
		
		return ret;
	}
	
	
	/**
	 * NOTE: if force is set, we set and recompile the respective indexing hops;
	 * otherwise, we release the forced exec type and recompile again. Hence, 
	 * any changes can be exactly reverted with the same access behavior.
	 * 
	 * @param sb
	 * @param pb
	 * @param var
	 * @param ec
	 * @param force
	 * @throws DMLUnsupportedOperationException
	 * @throws DMLRuntimeException
	 */
	public static void rFindAndRecompileIndexingHOP( StatementBlock sb, ProgramBlock pb, String var, ExecutionContext ec, boolean force )
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		if( pb instanceof IfProgramBlock && sb instanceof IfStatementBlock )
		{
			IfProgramBlock ipb = (IfProgramBlock) pb;
			IfStatementBlock isb = (IfStatementBlock) sb;
			IfStatement is = (IfStatement) sb.getStatement(0);
			//process if condition
			if( isb.getPredicateHops()!=null )
				ipb.setPredicate( rFindAndRecompileIndexingHOP(isb.getPredicateHops(),ipb.getPredicate(),var,ec,force) );
			//process if branch
			int len = is.getIfBody().size(); 
			for( int i=0; i<ipb.getChildBlocksIfBody().size() && i<len; i++ )
			{
				ProgramBlock lpb = ipb.getChildBlocksIfBody().get(i);
				StatementBlock lsb = is.getIfBody().get(i);
				rFindAndRecompileIndexingHOP(lsb,lpb,var,ec,force);
			}
			//process else branch
			if( ipb.getChildBlocksElseBody() != null )
			{
				int len2 = is.getElseBody().size();
				for( int i=0; i<ipb.getChildBlocksElseBody().size() && i<len2; i++ )
				{
					ProgramBlock lpb = ipb.getChildBlocksElseBody().get(i);
					StatementBlock lsb = is.getElseBody().get(i);
					rFindAndRecompileIndexingHOP(lsb,lpb,var,ec,force);
				}
			}				
		}
		else if( pb instanceof WhileProgramBlock && sb instanceof WhileStatementBlock )
		{
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
		else if( pb instanceof ForProgramBlock && sb instanceof ForStatementBlock ) //for or parfor
		{
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
			for( int i=0; i<fpb.getChildBlocks().size() && i<len; i++ )
			{
				ProgramBlock lpb = fpb.getChildBlocks().get(i);
				StatementBlock lsb = fs.getBody().get(i);
				rFindAndRecompileIndexingHOP(lsb,lpb,var,ec, force);
			}	
		}
		else //last level program block
		{
			try
			{
				//process actual hops
				boolean ret = false;
				Hop.resetVisitStatus(sb.get_hops());
				if( force )
				{
					//set forced execution type
					for( Hop h : sb.get_hops() )
						ret |= rFindAndSetCPIndexingHOP(h, var);
				}
				else
				{
					//release forced execution type
					for( Hop h : sb.get_hops() )
						ret |= rFindAndReleaseIndexingHOP(h, var);
				}
				
				//recompilation on-demand
				if( ret )
				{
					//construct new instructions
					ArrayList<Instruction> newInst = Recompiler.recompileHopsDag(sb, sb.get_hops(), ec.getVariables(), null, true, 0);
					pb.setInstructions( newInst ); 
				}
			}
			catch(Exception ex)
			{
				throw new DMLRuntimeException(ex);
			}
		}
	}
	
	/**
	 * 
	 * @param prog
	 * @param parforSB
	 * @param vars
	 * @return
	 * @throws DMLUnsupportedOperationException
	 * @throws DMLRuntimeException
	 */
	public static LocalVariableMap getReusableScalarVariables( DMLProgram prog, StatementBlock parforSB, LocalVariableMap vars ) 
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{
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
		throws DMLUnsupportedOperationException, DMLRuntimeException, HopsException
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
			ArrayList<Hop> hops = sb.get_hops();
			if( hops != null ) 
			{	
				//replace constant literals
				Hop.resetVisitStatus(hops);
				for( Hop hopRoot : hops )
					Recompiler.rReplaceLiterals( hopRoot, vars );
			}	
		}
	}
	
	/**
	 * 
	 * @param pred
	 * @param vars
	 * @throws DMLRuntimeException
	 */
	private static void replacePredicateLiterals( Hop pred, LocalVariableMap vars )
		throws DMLRuntimeException
	{
		if( pred != null ){
			pred.resetVisitStatus();
			Recompiler.rReplaceLiterals(pred, vars);
		}
	}
	
	/**
	 * This function determines if an parfor input variable is guaranteed to be read-only
	 * across multiple invocations of parfor optimization (e.g., in a surrounding while loop).
	 * In case of invariant variables we can reuse partitioned matrices and propagate constants
	 * for better size estimation.
	 * 
	 * @param prog
	 * @param parforSB
	 * @param var
	 * @return
	 * @throws DMLUnsupportedOperationException
	 * @throws DMLRuntimeException
	 */
	public static boolean isApplicableForReuseVariable( DMLProgram prog, StatementBlock parforSB, String var )
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		boolean ret = false;
		
		for( StatementBlock sb : prog.getStatementBlocks() )
			ret |= isApplicableForReuseVariable(sb, parforSB, var);
		
		return  ret;	
	}
	
	/**
	 * 
	 * @param sb
	 * @param parforSB
	 * @param var
	 * @throws DMLUnsupportedOperationException
	 * @throws DMLRuntimeException
	 */
	private static boolean isApplicableForReuseVariable( StatementBlock sb, StatementBlock parforSB, String var )
			throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		boolean ret = false;
		
		if( sb instanceof IfStatementBlock )
		{
			IfStatement is = (IfStatement) sb.getStatement(0);
			for( StatementBlock lsb : is.getIfBody() )
				ret |= isApplicableForReuseVariable(lsb, parforSB, var);
			for( StatementBlock lsb : is.getElseBody() )
				ret |= isApplicableForReuseVariable(lsb, parforSB, var);
		}
		else if( sb instanceof WhileStatementBlock )
		{
			WhileStatement ws = (WhileStatement) sb.getStatement(0);
			for( StatementBlock lsb : ws.getBody() )
				ret |= isApplicableForReuseVariable(lsb, parforSB, var);		
		}
		else if( sb instanceof ForStatementBlock ) //for or parfor
		{
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
	
	/**
	 * 
	 * @param pb
	 * @return
	 */
	public static boolean containsAtLeastOneFunction( ProgramBlock pb )
	{
		if( pb instanceof IfProgramBlock )
		{
			IfProgramBlock ipb = (IfProgramBlock) pb;
			for( ProgramBlock lpb : ipb.getChildBlocksIfBody() )
				if( containsAtLeastOneFunction(lpb) )
					return true;
			for( ProgramBlock lpb : ipb.getChildBlocksElseBody() )
				if( containsAtLeastOneFunction(lpb) )
					return true;
		}
		else if( pb instanceof WhileProgramBlock )
		{
			WhileProgramBlock wpb = (WhileProgramBlock) pb;
			for( ProgramBlock lpb : wpb.getChildBlocks() )
				if( containsAtLeastOneFunction(lpb) )
					return true;
		}
		else if( pb instanceof ForProgramBlock ) //incl parfor
		{
			ForProgramBlock fpb = (ForProgramBlock) pb;
			for( ProgramBlock lpb : fpb.getChildBlocks() )
				if( containsAtLeastOneFunction(lpb) )
					return true;
		}
		else
		{
			if( pb.getInstructions() != null )
				for( Instruction inst : pb.getInstructions() )
					if( inst instanceof FunctionCallCPInstruction )
						return true;
		}
		
		return false;	
	}
	
	/**
	 * 
	 * @param hop
	 * @param in
	 * @param force
	 * @return
	 * @throws DMLRuntimeException 
	 */
	private static ArrayList<Instruction> rFindAndRecompileIndexingHOP( Hop hop, ArrayList<Instruction> in, String var, ExecutionContext ec, boolean force ) 
		throws DMLRuntimeException
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
				tmp = Recompiler.recompileHopsDag(hop, ec.getVariables(), null, true, 0);
			}
		}
		catch(Exception ex)
		{
			throw new DMLRuntimeException(ex);
		}
		
		return tmp;
	}
	
	
	/**
	 * 
	 * @param hop
	 * @param var
	 * @return
	 */
	private static boolean rFindAndSetCPIndexingHOP(Hop hop, String var) 
	{
		boolean ret = false;
		
		if( hop.getVisited() == VisitStatus.DONE )
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
		
		hop.setVisited(VisitStatus.DONE);
		
		return ret;
	}
	
	private static boolean rFindAndReleaseIndexingHOP(Hop hop, String var) 
	{
		boolean ret = false;
		
		if( hop.getVisited() == VisitStatus.DONE )
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
		
		hop.setVisited(VisitStatus.DONE);
		
		return ret;
	}
	

	///////
	// additional general-purpose functionalities
	
	/**
	 * 
	 * @param iterVar
	 * @param offset
	 * @return
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException
	 */
	protected static ArrayList<Instruction> createNestedParallelismToInstructionSet(String iterVar, String offset) 
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{
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
		ArrayList<Instruction> tmp = new ArrayList<Instruction>();
		Instruction inst = ArithmeticBinaryCPInstruction.parseInstruction(str);
		tmp.add(inst);
		
		return tmp;
	}
	
	
	
	
	/////////////////////////////////
	// experimental functionality
	//////////
	
	/**
	 * 
	 * @param n
	 * @throws DMLRuntimeException
	 */
	protected static void recompilePartialPlan( OptNode n ) 
		throws DMLRuntimeException 
	{
		//NOTE: need to recompile complete programblock because (1) many to many relationships
		//between hops and instructions and (2) due to changed internal variable names 
		
		try
		{
			//get parent program and statement block
			OptTreePlanMappingAbstract map = OptTreeConverter.getAbstractPlanMapping();
			long pid = map.getMappedParentID(n.getID());
			Object[] o = map.getMappedProg(pid);
			StatementBlock sbOld = (StatementBlock) o[0];
			ProgramBlock pbOld = (ProgramBlock) o[1];
			
			//get changed node and set type appropriately
			Hop hop = (Hop) map.getMappedHop(n.getID());
			hop.setForcedExecType(n.getExecType().toLopsExecType()); 
			hop.setLops(null); //to enable fresh construction
		
			//get all hops of statement and construct new instructions
			Dag<Lop> dag = new Dag<Lop>();
			for( Hop hops : sbOld.get_hops() )
			{
				hops.resetVisitStatus();
				Recompiler.rClearLops(hops);
				Lop lops = hops.constructLops();
				lops.addToDag(dag);
			}
			
			//construct new instructions
			ArrayList<Instruction> newInst = dag.getJobs(sbOld, ConfigurationManager.getConfig());
			
			
			//exchange instructions
			pbOld.getInstructions().clear();
			pbOld.getInstructions().addAll(newInst);
		}
		catch(Exception ex)
		{
			throw new DMLRuntimeException(ex);
		}
	}

	
	/**
	 * NOTE: need to recompile complete programblock because (1) many to many relationships
	 * between hops and instructions and (2) due to changed internal variable names 
	 * 
	 * @param n
	 * @return
	 * @throws DMLRuntimeException
	 */
	protected static ProgramBlock recompile( OptNode n ) 
		throws DMLRuntimeException 
	{
		ProgramBlock pbNew = null;
		
		try
		{
			if( n.getNodeType() == NodeType.HOP )
			{
				//get parent program and statement block
				OptTreePlanMappingAbstract map = OptTreeConverter.getAbstractPlanMapping();
				long pid = map.getMappedParentID(n.getID());
				Object[] o = map.getMappedProg(pid);
				StatementBlock sbOld = (StatementBlock) o[0];
				ProgramBlock pbOld = (ProgramBlock) o[1];
				LopProperties.ExecType oldtype = null;
				
				//get changed node and set type appropriately
				Hop hop = (Hop) map.getMappedHop(n.getID());
				hop.setForcedExecType(n.getExecType().toLopsExecType()); 
				hop.setLops(null); //to enable fresh construction
			
				//get all hops of statement and construct new lops
				Dag<Lop> dag = new Dag<Lop>();
				for( Hop hops : sbOld.get_hops() )
				{
					hops.resetVisitStatus();
					Recompiler.rClearLops(hops);
					Lop lops = hops.constructLops();
					lops.addToDag(dag);
				}
				
				//construct new instructions
				ArrayList<Instruction> newInst = dag.getJobs(sbOld, ConfigurationManager.getConfig());
				
				//exchange instructions
				pbNew = new ProgramBlock(pbOld.getProgram());
				pbNew.setInstructions(newInst);
				
				//reset type global repository
				hop.setForcedExecType(oldtype);
				
			}
			else if( n.getNodeType() == NodeType.PARFOR )
			{	
				//no recompilation required
				OptTreePlanMappingAbstract map = OptTreeConverter.getAbstractPlanMapping();
				ParForProgramBlock pb = (ParForProgramBlock)map.getMappedProg(n.getID())[1];
				pbNew = ProgramConverter.createShallowCopyParForProgramBlock(pb, pb.getProgram());
				((ParForProgramBlock)pbNew).setExecMode(n.getExecType().toParForExecMode());
			}
			else
			{
				throw new DMLRuntimeException("Unexpected node type.");
			}
		}
		catch(Exception ex)
		{
			throw new DMLRuntimeException(ex);
		}
		
		return pbNew;
	}


	/**
	 * 
	 * @param hlNodeID
	 * @param pbNew
	 * @throws DMLRuntimeException
	 */
	protected static void exchangeProgram(long hlNodeID, ProgramBlock pbNew) 
		throws DMLRuntimeException 
	{
		OptTreePlanMappingAbstract map = OptTreeConverter.getAbstractPlanMapping();
		OptNode node = map.getOptNode(hlNodeID);
		
		if( node.getNodeType() == NodeType.HOP )
		{
			long pid = map.getMappedParentID(hlNodeID);
			Object[] o = map.getMappedProg(pid);
			ProgramBlock pbOld = (ProgramBlock) o[1];
			
			//exchange instructions (save version)
			pbOld.getInstructions().clear();
			pbOld.getInstructions().addAll( pbNew.getInstructions() );
		}
		else if( node.getNodeType() == NodeType.PARFOR )
		{
			ParForProgramBlock pbOld = (ParForProgramBlock) map.getMappedProg(node.getID())[1];
			pbOld.setExecMode(((ParForProgramBlock)pbNew).getExecMode());
			//TODO extend as required
		}
		else
		{
			throw new DMLRuntimeException("Unexpected node type: "+node.getNodeType());
		}
	}
	
}
