/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram.parfor.opt;

import java.util.ArrayList;

import com.ibm.bi.dml.conf.ConfigurationManager;
import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.hops.IndexingOp;
import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.hops.Hop.VISIT_STATUS;
import com.ibm.bi.dml.lops.LopProperties;
import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.lops.compile.Dag;
import com.ibm.bi.dml.lops.compile.Recompiler;
import com.ibm.bi.dml.parser.DMLProgram;
import com.ibm.bi.dml.parser.ForStatement;
import com.ibm.bi.dml.parser.ForStatementBlock;
import com.ibm.bi.dml.parser.IfStatement;
import com.ibm.bi.dml.parser.IfStatementBlock;
import com.ibm.bi.dml.parser.StatementBlock;
import com.ibm.bi.dml.parser.WhileStatement;
import com.ibm.bi.dml.parser.WhileStatementBlock;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.ForProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.IfProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.WhileProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.parfor.ProgramConverter;
import com.ibm.bi.dml.runtime.controlprogram.parfor.opt.OptNode.NodeType;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.ArithmeticBinaryCPInstruction;

/**
 * 
 */
public class ProgramRecompiler 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	
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
		if( pb instanceof IfProgramBlock )
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
		else if( pb instanceof WhileProgramBlock )
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
		else if( pb instanceof ForProgramBlock ) //for or parfor
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
					ArrayList<Instruction> newInst = Recompiler.recompileHopsDag(sb.get_hops(), ec.getVariables(), 0);
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
	 * @param var
	 * @return
	 * @throws DMLUnsupportedOperationException
	 * @throws DMLRuntimeException
	 */
	public static boolean isApplicableForReusePartitionedMatrix( DMLProgram prog, StatementBlock parforSB, String var )
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		boolean ret = false;
		
		for( StatementBlock sb : prog.getStatementBlocks() )
			ret |= isApplicableForReusePartitionedMatrix(sb, parforSB, var);
		
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
	private static boolean isApplicableForReusePartitionedMatrix( StatementBlock sb, StatementBlock parforSB, String var )
			throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		boolean ret = false;
		
		if( sb instanceof IfStatementBlock )
		{
			IfStatement is = (IfStatement) sb.getStatement(0);
			for( StatementBlock lsb : is.getIfBody() )
				ret |= isApplicableForReusePartitionedMatrix(lsb, parforSB, var);
			for( StatementBlock lsb : is.getElseBody() )
				ret |= isApplicableForReusePartitionedMatrix(lsb, parforSB, var);
		}
		else if( sb instanceof WhileStatementBlock )
		{
			WhileStatement ws = (WhileStatement) sb.getStatement(0);
			for( StatementBlock lsb : ws.getBody() )
				ret |= isApplicableForReusePartitionedMatrix(lsb, parforSB, var);		
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
					ret |= isApplicableForReusePartitionedMatrix(lsb, parforSB, var);
			}
		}
		
		return  ret && !sb.variablesUpdated().containsVariable(var);	
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
				tmp = Recompiler.recompileHopsDag(hop, ec.getVariables(), 0);
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
		
		if( hop.get_visited() == VISIT_STATUS.DONE )
			return ret;
		
		ArrayList<Hop> in = hop.getInput();
		
		if( hop instanceof IndexingOp )
		{
			String inMatrix = hop.getInput().get(0).get_name();
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
		
		hop.set_visited(VISIT_STATUS.DONE);
		
		return ret;
	}
	
	private static boolean rFindAndReleaseIndexingHOP(Hop hop, String var) 
	{
		boolean ret = false;
		
		if( hop.get_visited() == VISIT_STATUS.DONE )
			return ret;
		
		ArrayList<Hop> in = hop.getInput();
		
		if( hop instanceof IndexingOp )
		{
			String inMatrix = hop.getInput().get(0).get_name();
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
		
		hop.set_visited(VISIT_STATUS.DONE);
		
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
			hop.set_lops(null); //to enable fresh construction
		
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
			ArrayList<Instruction> newInst = dag.getJobs(ConfigurationManager.getConfig());
			
			
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
				hop.set_lops(null); //to enable fresh construction
			
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
				ArrayList<Instruction> newInst = dag.getJobs(ConfigurationManager.getConfig());
				
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
