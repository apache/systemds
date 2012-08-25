package com.ibm.bi.dml.runtime.controlprogram.parfor.opt;

import java.util.ArrayList;

import com.ibm.bi.dml.hops.Hops;
import com.ibm.bi.dml.hops.IndexingOp;
import com.ibm.bi.dml.lops.LopProperties;
import com.ibm.bi.dml.lops.Lops;
import com.ibm.bi.dml.lops.compile.Dag;
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
import com.ibm.bi.dml.runtime.controlprogram.parfor.opt.OptTreeConverter.HLObjectMapping;
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
			HLObjectMapping map = OptTreeConverter.getHLObjectMapping();
			long pid = map.getMappedParentID(n.getID());
			Object[] o = OptTreeConverter.getHLObjectMapping().getMappedProg(pid);
			StatementBlock sbOld = (StatementBlock) o[0];
			ProgramBlock pbOld = (ProgramBlock) o[1];
			
			//get changed node and set type appropriately
			Hops hop = (Hops) map.getMappedHop(n.getID());
			hop.setExecType(n.getExecType().toLopsExecType()); 
			hop.set_lops(null); //to enable fresh construction
		
			//get all hops of statement and construct new instructions
			ArrayList<Instruction> newInst = null;
			Dag<Lops> dag = new Dag<Lops>();
			for( Hops hops : sbOld.get_hops() )
			{
				System.out.println(hops.getOpString());
				rClearLops(hops);
				Lops lops = hops.constructLops();
				lops.addToDag(dag);
				newInst = dag.getJobs(false,ConfigurationManager.getConfig());
			}
			
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
	 * 
	 * @param hop
	 */
	private static void rClearLops( Hops hop )
	{
		//System.out.println(hop.getOpString());
		
		//preserve Treads to prevent wrong rmfilevar instructions
		if( hop.getOpString().equals("TRead") )
			return; 
		
		//clear all relevant lops to allow for recompilation
		hop.set_lops(null);
		for( Hops c : hop.getInput() )
			rClearLops(c);
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
				HLObjectMapping map = OptTreeConverter.getHLObjectMapping();
				long pid = map.getMappedParentID(n.getID());
				Object[] o = map.getMappedProg(pid);
				StatementBlock sbOld = (StatementBlock) o[0];
				ProgramBlock pbOld = (ProgramBlock) o[1];
				LopProperties.ExecType oldtype = null;
				
				//get changed node and set type appropriately
				Hops hop = (Hops) map.getMappedHop(n.getID());
				oldtype = hop.getExecType();
				System.out.println(hop.get_name());
				System.out.println(oldtype);
				System.out.println(n.getExecType());
				hop.setExecType(n.getExecType().toLopsExecType()); 
				hop.set_lops(null); //to enable fresh construction
			
				//get all hops of statement and construct new instructions
				ArrayList<Instruction> newInst = null;
				Dag<Lops> dag = new Dag<Lops>();
				for( Hops hops : sbOld.get_hops() )
				{
					rClearLops(hops);
					Lops lops = hops.constructLops();
					lops.addToDag(dag);
					newInst = dag.getJobs(false,ConfigurationManager.getConfig());
				}
				
				//exchange instructions
				pbNew = new ProgramBlock(pbOld.getProgram());
				pbNew.setInstructions(newInst);
				
				//reset type global repository
				hop.setExecType(oldtype);
				
			}
			else if( n.getNodeType() == NodeType.PARFOR )
			{	
				//no recompilation required
				HLObjectMapping map = OptTreeConverter.getHLObjectMapping();
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
		HLObjectMapping map = OptTreeConverter.getHLObjectMapping();
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
	
	public static ArrayList<Instruction> createNestedParallelismToInstructionSet(String iterVar, String offset) 
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{
		//create instruction string
		StringBuffer sb = new StringBuffer("CP"+Lops.OPERAND_DELIMITOR+"+"+Lops.OPERAND_DELIMITOR);
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
	

	public static void rFindAndRecompileIndexingHOP( StatementBlock sb, ProgramBlock pb, String var )
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		if( pb instanceof IfProgramBlock )
		{
			IfProgramBlock ipb = (IfProgramBlock) pb;
			IfStatement is = (IfStatement) sb.getStatement(0);
			
			for( int i=0; i<ipb.getChildBlocksIfBody().size(); i++ )
			{
				ProgramBlock lpb = ipb.getChildBlocksIfBody().get(0);
				StatementBlock lsb = is.getIfBody().get(0);
				rFindAndRecompileIndexingHOP(lsb,lpb,var);
			}
			//process else condition
			if( ipb.getChildBlocksElseBody() != null )
			{
				for( int i=0; i<ipb.getChildBlocksElseBody().size(); i++ )
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
			for( int i=0; i<wpb.getChildBlocks().size(); i++ )
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
			for( int i=0; i<fpb.getChildBlocks().size(); i++ )
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
					ret |= rFindAndSetCPIndexingHOP(h, var);
				}
				
				//recompilation on-demand
				if( ret )
				{
					//get all hops of statement and construct new instructions
					ArrayList<Instruction> newInst = null;
					Dag<Lops> dag = new Dag<Lops>();
					for( Hops hops : sb.get_hops() )
					{
						rClearLops(hops);
						Lops lops = hops.constructLops();
						lops.addToDag(dag);
						newInst = dag.getJobs(false,ConfigurationManager.getConfig());
					}
					
					//exchange instructions
					pb.getInstructions().clear();
					pb.getInstructions().addAll(newInst);
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
		ArrayList<Hops> in = hop.getInput();
		
		if( hop instanceof IndexingOp )
		{
			String inMatrix = hop.getInput().get(0).get_name();
			if( inMatrix.equals(var) )
			{
				hop.setExecType(LopProperties.ExecType.CP);
				hop.set_lops(null); //for fresh reconstruction
				ret = true;
			}
		}
		
		//recursive search
		if( in != null )
			for( Hops hin : in )
				ret |= rFindAndSetCPIndexingHOP(hin,var);
		
		return ret;
	}
}
