package com.ibm.bi.dml.runtime.controlprogram.parfor.opt;

import java.util.ArrayList;

import com.ibm.bi.dml.hops.Hops;
import com.ibm.bi.dml.hops.IndexingOp;
import com.ibm.bi.dml.hops.Hops.VISIT_STATUS;
import com.ibm.bi.dml.lops.LopProperties;
import com.ibm.bi.dml.lops.Lops;
import com.ibm.bi.dml.lops.compile.Dag;
import com.ibm.bi.dml.lops.compile.Recompiler;
import com.ibm.bi.dml.parser.ForStatement;
import com.ibm.bi.dml.parser.ForStatementBlock;
import com.ibm.bi.dml.parser.IfStatement;
import com.ibm.bi.dml.parser.StatementBlock;
import com.ibm.bi.dml.parser.WhileStatement;
import com.ibm.bi.dml.runtime.controlprogram.ForProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.IfProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.WhileProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.parfor.ProgramConverter;
import com.ibm.bi.dml.runtime.controlprogram.parfor.opt.OptNode.NodeType;
import com.ibm.bi.dml.runtime.controlprogram.parfor.opt.OptNode.ParamType;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.ConfigurationManager;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.ArithmeticBinaryCPInstruction;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.DMLUnsupportedOperationException;

/**
 * 
 */
public class ProgramRecompiler 
{
	/**
	 * 
	 * @param n
	 * @throws DMLRuntimeException
	 */
	public static void recompilePartialPlan( OptNode n ) 
		throws DMLRuntimeException 
	{
		System.out.println("recompiling "+n.getParam(ParamType.OPSTRING));
		
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
			Hops hop = (Hops) map.getMappedHop(n.getID());
			hop.setForcedExecType(n.getExecType().toLopsExecType()); 
			hop.set_lops(null); //to enable fresh construction
		
			//get all hops of statement and construct new instructions
			Dag<Lops> dag = new Dag<Lops>();
			for( Hops hops : sbOld.get_hops() )
			{
				hops.resetVisitStatus();
				Recompiler.rClearLops(hops);
				Lops lops = hops.constructLops();
				lops.addToDag(dag);
			}
			
			//construct new instructions
			ArrayList<Instruction> newInst = dag.getJobs(ConfigurationManager.getConfig());
			
			
			//exchange instructions
			System.out.println("OLD");
			System.out.println(pbOld.getInstructions());
			pbOld.getInstructions().clear();
			
			pbOld.getInstructions().addAll(newInst);
			System.out.println("NEW");
			System.out.println(pbOld.getInstructions());
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
	public static ProgramBlock recompile( OptNode n ) 
		throws DMLRuntimeException 
	{
		//System.out.println("recompiling "+n.getParam(ParamType.OPSTRING));
		
		
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
				Hops hop = (Hops) map.getMappedHop(n.getID());
				hop.setForcedExecType(n.getExecType().toLopsExecType()); 
				hop.set_lops(null); //to enable fresh construction
			
				//get all hops of statement and construct new lops
				Dag<Lops> dag = new Dag<Lops>();
				for( Hops hops : sbOld.get_hops() )
				{
					hops.resetVisitStatus();
					Recompiler.rClearLops(hops);
					Lops lops = hops.constructLops();
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
	public static void exchangeProgram(long hlNodeID, ProgramBlock pbNew) 
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
	public static ArrayList<Instruction> createNestedParallelismToInstructionSet(String iterVar, String offset) 
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{
		//create instruction string
		StringBuilder sb = new StringBuilder("CP"+Lops.OPERAND_DELIMITOR+"+"+Lops.OPERAND_DELIMITOR);
		sb.append(iterVar);
		sb.append(Lops.DATATYPE_PREFIX+"SCALAR"+Lops.VALUETYPE_PREFIX+"INT"+Lops.OPERAND_DELIMITOR);
		sb.append(offset);
		sb.append(Lops.DATATYPE_PREFIX+"SCALAR"+Lops.VALUETYPE_PREFIX+"INT"+Lops.OPERAND_DELIMITOR);
		sb.append(iterVar);
		sb.append(Lops.DATATYPE_PREFIX+"SCALAR"+Lops.VALUETYPE_PREFIX+"INT");
		String str = sb.toString(); 
		
		//create instruction set
		ArrayList<Instruction> tmp = new ArrayList<Instruction>();
		Instruction inst = ArithmeticBinaryCPInstruction.parseInstruction(str);
		tmp.add(inst);
		
		return tmp;
	}
	
	/**
	 * 
	 * @param sb
	 * @param pb
	 * @param var
	 * @throws DMLUnsupportedOperationException
	 * @throws DMLRuntimeException
	 */
	public static void rFindAndRecompileIndexingHOP( StatementBlock sb, ProgramBlock pb, String var )
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		if( pb instanceof IfProgramBlock )
		{
			IfProgramBlock ipb = (IfProgramBlock) pb;
			IfStatement is = (IfStatement) sb.getStatement(0);
			
			int len = is.getIfBody().size(); //robustness for potentially added problem blocks
			for( int i=0; i<ipb.getChildBlocksIfBody().size() && i<len; i++ )
			{
				ProgramBlock lpb = ipb.getChildBlocksIfBody().get(0);
				StatementBlock lsb = is.getIfBody().get(0);
				rFindAndRecompileIndexingHOP(lsb,lpb,var);
			}
			//process else condition
			if( ipb.getChildBlocksElseBody() != null )
			{
				int len2 = is.getElseBody().size();
				for( int i=0; i<ipb.getChildBlocksElseBody().size() && i<len2; i++ )
				{
					ProgramBlock lpb = ipb.getChildBlocksElseBody().get(i);
					StatementBlock lsb = is.getElseBody().get(i);
					rFindAndRecompileIndexingHOP(lsb,lpb,var);
				}
			}				
		}
		else if( pb instanceof WhileProgramBlock )
		{
			WhileProgramBlock wpb = (WhileProgramBlock) pb;
			WhileStatement ws = (WhileStatement) sb.getStatement(0);
			//process body
			int len = ws.getBody().size(); //robustness for potentially added problem blocks
			for( int i=0; i<wpb.getChildBlocks().size() && i<len; i++ )
			{
				ProgramBlock lpb = wpb.getChildBlocks().get(i);
				StatementBlock lsb = ws.getBody().get(i);
				rFindAndRecompileIndexingHOP(lsb,lpb,var);
			}			
		}
		else if( pb instanceof ForProgramBlock ) //for or parfor
		{
			ForProgramBlock fpb = (ForProgramBlock) pb;
			ForStatementBlock fsb = (ForStatementBlock)sb;
			ForStatement fs = (ForStatement) fsb.getStatement(0);
			//process body
			int len = fs.getBody().size(); //robustness for potentially added problem blocks
			for( int i=0; i<fpb.getChildBlocks().size() && i<len; i++ )
			{
				ProgramBlock lpb = fpb.getChildBlocks().get(i);
				StatementBlock lsb = fs.getBody().get(i);
				rFindAndRecompileIndexingHOP(lsb,lpb,var);
			}	
		}
		else //last level program block
		{
			try
			{
				//process actual hops
				boolean ret = false;
				for( Hops h : sb.get_hops() )
				{
					h.resetVisitStatus();
					ret |= rFindAndSetCPIndexingHOP(h, var);
				}
				
				//recompilation on-demand
				if( ret )
				{
					//construct new instructions
					ArrayList<Instruction> newInst = Recompiler.recompileHopsDag(sb.get_hops(), pb.getVariables(), 0);
					pb.setInstructions( newInst );   
				}
			}
			catch(Exception ex)
			{
				throw new DMLRuntimeException(ex);
			}
		}
	}
	
	public static boolean rFindAndSetCPIndexingHOP(Hops hop, String var) 
	{
		boolean ret = false;
		
		if( hop.get_visited() == VISIT_STATUS.DONE )
			return ret;
		
		ArrayList<Hops> in = hop.getInput();
		
		if( hop instanceof IndexingOp )
		{
			String inMatrix = hop.getInput().get(0).get_name();
			if( inMatrix.equals(var) )
			{
				//NOTE: mem estimate of RIX, set to output size by parfor optmizer
				//(rowblock/colblock only applied if in total less than two blocks,
				// hence always mem_est<mem_budget)
				if( hop.getMemEstimate() < Hops.getMemBudget(true) )
					hop.setForcedExecType( LopProperties.ExecType.CP );
				else
					hop.setForcedExecType( LopProperties.ExecType.CP_FILE );
				
				ret = true;
			}
		}
		
		//recursive search
		if( in != null )
			for( Hops hin : in )
				ret |= rFindAndSetCPIndexingHOP(hin,var);
		
		hop.set_visited(VISIT_STATUS.DONE);
		
		return ret;
	}
}
