package com.ibm.bi.dml.runtime.controlprogram.parfor.opt;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map.Entry;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.hops.DataOp;
import com.ibm.bi.dml.hops.Hops;
import com.ibm.bi.dml.hops.LiteralOp;
import com.ibm.bi.dml.parser.ForStatement;
import com.ibm.bi.dml.parser.ForStatementBlock;
import com.ibm.bi.dml.parser.IfStatement;
import com.ibm.bi.dml.parser.ParForStatement;
import com.ibm.bi.dml.parser.ParForStatementBlock;
import com.ibm.bi.dml.parser.StatementBlock;
import com.ibm.bi.dml.parser.WhileStatement;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.runtime.controlprogram.ForProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.IfProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.LocalVariableMap;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.WhileProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.parfor.opt.OptNode.ExecType;
import com.ibm.bi.dml.runtime.controlprogram.parfor.opt.OptNode.NodeType;
import com.ibm.bi.dml.runtime.controlprogram.parfor.opt.OptNode.ParamType;
import com.ibm.bi.dml.runtime.controlprogram.parfor.opt.PerfTestTool.DataFormat;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.IDSequence;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.ComputationCPInstruction;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.Data;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.FunctionCallCPInstruction;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.MatrixObjectNew;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.RandCPInstruction;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.MatrixFormatMetaData;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlockDSM;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.DMLUnsupportedOperationException;
import com.ibm.bi.dml.utils.HopsException;

/**
 * Converter for creating an internal plan representation for a given runtime program
 * and to modify/create the runtime program according to the optimized plan.
 * 
 * 
 */
public class OptTreeConverter 
{	
	//current single plan (hl, rt, stats)
	private static HLObjectMapping  _hlObjMap;
	private static RTObjectMapping  _rtObjMap;
	
	private static OptNode _tmpParent   = null;
	private static OptNode _tmpChildOld = null;
	private static OptNode _tmpChildNew = null;
	
	static
	{
		_hlObjMap = new HLObjectMapping();
		_rtObjMap = new RTObjectMapping();
	}
	
	/**
	 * 
	 * @param ck
	 * @param cm
	 * @param pfpb
	 * @return
	 * @throws DMLUnsupportedOperationException
	 * @throws DMLRuntimeException
	 */
	public static OptTree createOptTree( int ck, double cm, ParForProgramBlock pfpb ) 
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		OptNode root = rCreateOptNode( pfpb, pfpb.getVariables(), true, true );		
		OptTree tree = new OptTree(ck, cm, root);
		
		if( DMLScript.DEBUG )
			System.out.println( tree.explain(true) );
			
		return tree;
	}
	
	public static OptTree createAbstractOptTree( int ck, double cm, ParForStatementBlock pfsb, ParForProgramBlock pfpb ) 
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		OptTree tree = null;
		OptNode root = null;
		
		try
		{
			root = rCreateAbstractOptNode( pfsb, pfpb, pfpb.getVariables(), true );
			tree = new OptTree(ck, cm, root);
		}
		catch(HopsException he)
		{
			throw new DMLRuntimeException(he);
		}	
		
		if( DMLScript.DEBUG )
			System.out.println( tree.explain(true) );
			
		return tree;
	}

	/**
	 * 
	 * @param pb
	 * @param vars
	 * @param topLevel
	 * @return
	 * @throws DMLUnsupportedOperationException
	 * @throws DMLRuntimeException
	 */
	public static OptNode rCreateOptNode( ProgramBlock pb, LocalVariableMap vars, boolean topLevel, boolean storeObjs ) 
		throws DMLUnsupportedOperationException, DMLRuntimeException 
	{
		OptNode node = null;
		
		if( pb instanceof IfProgramBlock )
		{
			IfProgramBlock ipb = (IfProgramBlock) pb;
			node = new OptNode( NodeType.IF );
			if(storeObjs)
				_rtObjMap.putMapping(ipb, node);
			node.setExecType(ExecType.CP);
			//process if condition
			OptNode ifn = new OptNode(NodeType.GENERIC);
			node.addChilds( createOptNodes( ipb.getPredicate(), vars,storeObjs ) );
			node.addChild( ifn );
			for( ProgramBlock lpb : ipb.getChildBlocksIfBody() )
				ifn.addChild( rCreateOptNode(lpb,vars,topLevel, storeObjs) );
			//process else condition
			if( ipb.getChildBlocksElseBody() != null )
			{
				OptNode efn = new OptNode(NodeType.GENERIC);
				node.addChild( efn );
				for( ProgramBlock lpb : ipb.getChildBlocksElseBody() )
					efn.addChild( rCreateOptNode(lpb,vars,topLevel, storeObjs) );
			}				
		}
		else if( pb instanceof WhileProgramBlock )
		{
			WhileProgramBlock wpb = (WhileProgramBlock) pb;
			node = new OptNode( NodeType.WHILE );
			if(storeObjs)
				_rtObjMap.putMapping(wpb, node);
			node.setExecType(ExecType.CP);
			//process predicate instruction
			node.addChilds( createOptNodes( wpb.getPredicate(), vars,storeObjs ) );
			//process body
			for( ProgramBlock lpb : wpb.getChildBlocks() )
				node.addChild( rCreateOptNode(lpb,vars,topLevel,storeObjs) );
			
		}
		else if( pb instanceof ForProgramBlock && !(pb instanceof ParForProgramBlock) )
		{
			ForProgramBlock fpb = (ForProgramBlock) pb;
			node = new OptNode( NodeType.FOR );
			if(storeObjs)
				_rtObjMap.putMapping(fpb, node);
			node.setExecType(ExecType.CP);
			
			node.addParam(ParamType.NUM_ITERATIONS, String.valueOf(CostEstimator.FACTOR_NUM_ITERATIONS));
			
			node.addChilds( createOptNodes( fpb.getFromInstructions(), vars,storeObjs ) );
			node.addChilds( createOptNodes( fpb.getToInstructions(), vars,storeObjs ) );
			node.addChilds( createOptNodes( fpb.getIncrementInstructions(), vars,storeObjs ) );
			
			//process body
			for( ProgramBlock lpb : fpb.getChildBlocks() )
				node.addChild( rCreateOptNode(lpb,vars,topLevel,storeObjs) );
		}
		else if( pb instanceof ParForProgramBlock )
		{
			ParForProgramBlock fpb = (ParForProgramBlock) pb;			
			node = new OptNode( NodeType.PARFOR );
			if(storeObjs)
				_rtObjMap.putMapping(fpb, node);
			node.setK( fpb.getDegreeOfParallelism() );
			int N = fpb.getNumIterations();
			node.addParam(ParamType.NUM_ITERATIONS, (N!=-1) ? String.valueOf(N) : 
															  String.valueOf(CostEstimator.FACTOR_NUM_ITERATIONS));
			
			switch(fpb.getExecMode())
			{
				case LOCAL:
					node.setExecType(ExecType.CP);
					break;
				case REMOTE_MR:
					node.setExecType(ExecType.MR);
					break;
			}
			
			if( !topLevel )
			{
				node.addChilds( createOptNodes( fpb.getFromInstructions(), vars, storeObjs ) );
				node.addChilds( createOptNodes( fpb.getToInstructions(), vars, storeObjs ) );
				node.addChilds( createOptNodes( fpb.getIncrementInstructions(), vars, storeObjs ) );
			}
			
			//process body
			for( ProgramBlock lpb : fpb.getChildBlocks() )
				node.addChild( rCreateOptNode(lpb,vars,false,storeObjs) );			
			
			//parameters, add required parameters
		}
		else //last level program block
		{
			node = new OptNode(NodeType.GENERIC);
			if(storeObjs)
				_rtObjMap.putMapping(pb, node);
			node.addChilds( createOptNodes(pb.getInstructions(), vars, storeObjs) );
			node.setExecType(ExecType.CP);
		}
			
		return node;
	}
	


	/**
	 * 
	 * @param instset
	 * @param vars
	 * @return
	 * @throws DMLUnsupportedOperationException
	 * @throws DMLRuntimeException
	 */
	public static ArrayList<OptNode> createOptNodes (ArrayList<Instruction> instset, LocalVariableMap vars, boolean storeObjs) 
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		ArrayList<OptNode> tmp = new ArrayList<OptNode>(instset.size());
		for( Instruction inst : instset )
			tmp.add( createOptNode(inst,vars,storeObjs) );
		return tmp;
	}
	
	/**
	 * 
	 * @param inst
	 * @param vars
	 * @return
	 * @throws DMLUnsupportedOperationException
	 * @throws DMLRuntimeException
	 */
	private static OptNode createOptNode( Instruction inst, LocalVariableMap vars, boolean storeObjs ) 
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		OptNode node = new OptNode(NodeType.INST);
		String instStr = inst.toString();
		String opstr = instStr.split(Instruction.OPERAND_DELIM)[1];
		if(storeObjs)
			_rtObjMap.putMapping(inst, node);
		node.addParam(ParamType.OPSTRING,opstr);
		
		//exec type
		switch( inst.getType() )
		{
			case CONTROL_PROGRAM:
				node.setExecType(ExecType.CP);
				//exec operations
				//CPInstruction cpinst = (CPInstruction) inst;
				//node.addParam(ParamType.OPTYPE,cpinst.getCPInstructionType().toString());
				break;
			case MAPREDUCE:
			case MAPREDUCE_JOB:
				node.setExecType(ExecType.MR);
				//exec operations
				//MRInstruction mrinst = (MRInstruction) inst;
				//node.addParam(ParamType.OPTYPE,mrinst.getMRInstructionType().toString());
				break;
			default:
				throw new DMLUnsupportedOperationException("Unsupported instruction type.");
		}
		
		//create statistics 
		OptNodeStatistics stats = analyzeStatistics(inst, node, vars);
		node.setStatistics(stats);
		
		return node;
	}
	
	/**
	 * 
	 * @param sb
	 * @param pb
	 * @param vars
	 * @param topLevel
	 * @return
	 * @throws DMLUnsupportedOperationException
	 * @throws DMLRuntimeException
	 * @throws HopsException
	 */
	public static OptNode rCreateAbstractOptNode( StatementBlock sb, ProgramBlock pb, LocalVariableMap vars, boolean topLevel ) 
		throws DMLUnsupportedOperationException, DMLRuntimeException, HopsException 
	{
		OptNode node = null;
		
		if( pb instanceof IfProgramBlock )
		{
			IfProgramBlock ipb = (IfProgramBlock) pb;
			IfStatement is = (IfStatement) sb.getStatement(0);
			
			node = new OptNode( NodeType.IF );
			_hlObjMap.putProgMapping(sb, pb, node);
			node.setExecType(ExecType.CP);
			//process if condition
			OptNode ifn = new OptNode(NodeType.GENERIC);
			node.addChild( ifn );
			for( int i=0; i<ipb.getChildBlocksIfBody().size(); i++ )
			{
				ProgramBlock lpb = ipb.getChildBlocksIfBody().get(0);
				StatementBlock lsb = is.getIfBody().get(0);
				ifn.addChild( rCreateAbstractOptNode(lsb,lpb,vars,topLevel) );
			}
			//process else condition
			if( ipb.getChildBlocksElseBody() != null )
			{
				OptNode efn = new OptNode(NodeType.GENERIC);
				node.addChild( efn );
				for( int i=0; i<ipb.getChildBlocksElseBody().size(); i++ )
				{
					ProgramBlock lpb = ipb.getChildBlocksElseBody().get(i);
					StatementBlock lsb = is.getElseBody().get(i);
					ifn.addChild( rCreateAbstractOptNode(lsb,lpb,vars,topLevel) );
				}
			}				
		}
		else if( pb instanceof WhileProgramBlock )
		{
			WhileProgramBlock wpb = (WhileProgramBlock) pb;
			WhileStatement ws = (WhileStatement) sb.getStatement(0);
			
			node = new OptNode( NodeType.WHILE );
			_hlObjMap.putProgMapping(sb, pb, node);
			node.setExecType(ExecType.CP);
			//process body
			for( int i=0; i<wpb.getChildBlocks().size(); i++ )
			{
				ProgramBlock lpb = wpb.getChildBlocks().get(i);
				StatementBlock lsb = ws.getBody().get(i);
				node.addChild( rCreateAbstractOptNode(lsb,lpb,vars,topLevel) );
			}			
		}
		else if( pb instanceof ForProgramBlock && !(pb instanceof ParForProgramBlock) )
		{
			ForProgramBlock fpb = (ForProgramBlock) pb;
			ForStatementBlock fsb = (ForStatementBlock)sb;
			ForStatement fs = (ForStatement) fsb.getStatement(0);
			
			node = new OptNode( NodeType.FOR );
			_hlObjMap.putProgMapping(sb, pb, node);
			node.setExecType(ExecType.CP);
			
			node.addParam(ParamType.NUM_ITERATIONS, String.valueOf(CostEstimator.FACTOR_NUM_ITERATIONS));
			
			node.addChilds( rCreateAbstractOptNodes( fsb.getFromHops(), vars ) );
			node.addChilds( rCreateAbstractOptNodes( fsb.getToHops(), vars ) );
			node.addChilds( rCreateAbstractOptNodes( fsb.getIncrementHops(), vars ) );
			
			//process body
			for( int i=0; i<fpb.getChildBlocks().size(); i++ )
			{
				ProgramBlock lpb = fpb.getChildBlocks().get(i);
				StatementBlock lsb = fs.getBody().get(i);
				node.addChild( rCreateAbstractOptNode(lsb,lpb,vars,topLevel) );
			}	
		}
		else if( pb instanceof ParForProgramBlock )
		{
			ParForProgramBlock fpb = (ParForProgramBlock) pb;		
			ParForStatementBlock fsb = (ParForStatementBlock)sb;
			ParForStatement fs = (ParForStatement) fsb.getStatement(0);
			node = new OptNode( NodeType.PARFOR );
			_hlObjMap.putProgMapping(sb, pb, node);
			node.setK( fpb.getDegreeOfParallelism() );
			int N = fpb.getNumIterations();
			node.addParam(ParamType.NUM_ITERATIONS, (N!=-1) ? String.valueOf(N) : 
															  String.valueOf(CostEstimator.FACTOR_NUM_ITERATIONS));
			
			switch(fpb.getExecMode())
			{
				case LOCAL:
					node.setExecType(ExecType.CP);
					break;
				case REMOTE_MR:
					node.setExecType(ExecType.MR);
					break;
			}
			
			if( !topLevel )
			{
				node.addChilds( rCreateAbstractOptNodes( fsb.getFromHops(), vars ) );
				node.addChilds( rCreateAbstractOptNodes( fsb.getToHops(), vars ) );
				node.addChilds( rCreateAbstractOptNodes( fsb.getIncrementHops(), vars ) );
			}
			
			//process body
			for( int i=0; i<fpb.getChildBlocks().size(); i++ )
			{
				ProgramBlock lpb = fpb.getChildBlocks().get(i);
				StatementBlock lsb = fs.getBody().get(i);
				node.addChild( rCreateAbstractOptNode(lsb,lpb,vars,topLevel) );
			}
			
			//parameters, add required parameters
		}
		else //last level program block
		{
			node = new OptNode(NodeType.GENERIC);
			_hlObjMap.putProgMapping(sb, pb, node);
			node.addChilds( createAbstractOptNodes(sb.get_hops(), vars) );
			node.setExecType(ExecType.CP);
		}
			
		//TODO function call statement block
		
		return node;
	}

	//TODO predicate hops e.g., at whilestatementblock


	/**
	 * 
	 * @param hops
	 * @param vars
	 * @return
	 */
	public static ArrayList<OptNode> createAbstractOptNodes(ArrayList<Hops> hops, LocalVariableMap vars) 
	{
		ArrayList<OptNode> ret = new ArrayList<OptNode>(); 
		for( Hops hop : hops )
			ret.addAll(rCreateAbstractOptNodes(hop,vars));
		return ret;
	}
	
	/**
	 * 
	 * @param hop
	 * @param vars
	 * @return
	 */
	public static ArrayList<OptNode> rCreateAbstractOptNodes(Hops hop, LocalVariableMap vars) 
	{
		//System.out.println(hop.getOpString());
		
		ArrayList<OptNode> ret = new ArrayList<OptNode>(); 
		ArrayList<Hops> in = hop.getInput();
		
		if( !(hop instanceof DataOp || hop instanceof LiteralOp) )
		{
			OptNode node = new OptNode(NodeType.HOP);
			String opstr = hop.getOpString();
			_hlObjMap.putHopMapping(hop, node);
			node.addParam(ParamType.OPSTRING,opstr);
			ret.add(node);
		}
		
		if( in != null )
			for( Hops hin : in )
				if( !(hin instanceof DataOp || hin instanceof LiteralOp ) ) //no need for opt nodes
					ret.addAll(rCreateAbstractOptNodes(hin,vars));
		
		return ret;
	}

	
	
	/**
	 * 
	 * @param inst
	 * @param on
	 * @param vars
	 * @return
	 * @throws DMLRuntimeException
	 */
	private static OptNodeStatistics analyzeStatistics(Instruction inst, OptNode on, LocalVariableMap vars) 
		throws DMLRuntimeException 
	{
		OptNodeStatistics ret = null;
		String instName = on.getInstructionName();
		
		if( PerfTestTool.isRegisteredInstruction(instName) )
		{	
			if( inst instanceof RandCPInstruction )
			{
				RandCPInstruction linst = (RandCPInstruction) inst;
				DataFormat df = (linst.sparsity>MatrixBlockDSM.SPARCITY_TURN_POINT) ? 
						                        DataFormat.DENSE : DataFormat.SPARSE; 
				ret = new OptNodeStatistics(linst.rows, linst.cols, -1, -1, linst.sparsity, df);
			}
			else if ( inst instanceof FunctionCallCPInstruction )
			{
				FunctionCallCPInstruction linst = (FunctionCallCPInstruction)inst;
				ArrayList<String> params = linst.getBoundInputParamNames();
				ret = new OptNodeStatistics(); //default vals
				
				double maxSize = 0;
				for( String param : params ) //use the largest input matrix
				{
					Data dat = vars.get(param);
					if( dat!=null && dat.getDataType()==DataType.MATRIX )
					{
						MatrixObjectNew mdat1 = (MatrixObjectNew) dat;
						MatrixCharacteristics mc1 = ((MatrixFormatMetaData)mdat1.getMetaData()).getMatrixCharacteristics();
						
						if( mc1.numRows*mc1.numColumns > maxSize )
						{
							ret.setDim1( mc1.numRows );
							ret.setDim2( mc1.numColumns );
							ret.setSparsity( mc1.nonZero /(  ret.getDim1() * ret.getDim2() ) ); //sparsity
							ret.setDataFormat((ret.getSparsity() < MatrixBlockDSM.SPARCITY_TURN_POINT )? DataFormat.SPARSE : DataFormat.DENSE ); 
							maxSize = mc1.numRows*mc1.numColumns;
						}
					}
				}
			}
			else if ( inst instanceof ComputationCPInstruction ) //needs to be last CP case
			{
				//AggregateBinaryCPInstruction, AggregateUnaryCPInstruction, 
				//FunctionCallCPInstruction, ReorgCPInstruction
				
				ComputationCPInstruction linst = (ComputationCPInstruction) inst;
				ret = new OptNodeStatistics(); //default
				
				if( linst.input1 != null && linst.input2 != null ) //binary
				{
					Data dat1 = vars.get( linst.input1.get_name() );
					Data dat2 = vars.get( linst.input2.get_name() );
					
					if( dat1 != null )
					{
						MatrixObjectNew mdat1 = (MatrixObjectNew) dat1;
						MatrixCharacteristics mc1 = ((MatrixFormatMetaData)mdat1.getMetaData()).getMatrixCharacteristics();
						ret.setDim1( mc1.numRows );
						ret.setDim2( mc1.numColumns );
						ret.setSparsity( mc1.nonZero /( ret.getDim1() * ret.getDim2() ) ); //sparsity
						ret.setDataFormat((ret.getSparsity() < MatrixBlockDSM.SPARCITY_TURN_POINT )? DataFormat.SPARSE : DataFormat.DENSE); 
					}
					if( dat2 != null )
					{
						MatrixObjectNew mdat2 = (MatrixObjectNew) dat2;
						MatrixCharacteristics mc2 = ((MatrixFormatMetaData)mdat2.getMetaData()).getMatrixCharacteristics();
						ret.setDim3( mc2.numRows );
						ret.setDim4( mc2.numColumns );
						ret.setDataFormat( (ret.getSparsity() < MatrixBlockDSM.SPARCITY_TURN_POINT ) ? DataFormat.SPARSE : DataFormat.DENSE ); 
					}
				}
				else //unary
				{
					Data dat1 = vars.get( linst.input1.get_name() );
					
					if( dat1 != null )
					{
						MatrixObjectNew mdat1 = (MatrixObjectNew) dat1;
						MatrixCharacteristics mc1 = ((MatrixFormatMetaData)mdat1.getMetaData()).getMatrixCharacteristics();
						ret.setDim1( mc1.numRows );
						ret.setDim2( mc1.numColumns );
						ret.setSparsity( mc1.nonZero /( ret.getDim1() * ret.getDim2() ) ); //sparsity
						ret.setDataFormat((ret.getSparsity() < MatrixBlockDSM.SPARCITY_TURN_POINT ) ? DataFormat.SPARSE : DataFormat.DENSE); 
					}					
				}
			}
		}
		
		if( ret == null )
			ret = new OptNodeStatistics(); //default values
		
		return ret; //null if not reqistered for profiling
	}

	/**
	 * 
	 * @param parent
	 * @param n
	 * @param pbOld
	 * @param pbNew
	 * @throws DMLUnsupportedOperationException
	 */
	public static void replaceProgramBlock(OptNode parent, OptNode n, ProgramBlock pbOld, ProgramBlock pbNew) 
		throws DMLUnsupportedOperationException
	{
		ProgramBlock pbParent = (ProgramBlock)_rtObjMap.getMappedObject( parent.getID() );
		
		if( pbParent instanceof IfProgramBlock )
		{
			IfProgramBlock ipb = (IfProgramBlock) pbParent;
			replaceProgramBlock( ipb.getChildBlocksIfBody(), pbOld, pbNew );
			replaceProgramBlock( ipb.getChildBlocksElseBody(), pbOld, pbNew );				
		}
		else if( pbParent instanceof WhileProgramBlock )
		{
			WhileProgramBlock wpb = (WhileProgramBlock) pbParent;
			replaceProgramBlock( wpb.getChildBlocks(), pbOld, pbNew );			
		}
		else if( pbParent instanceof ForProgramBlock || pbParent instanceof ParForProgramBlock )
		{
			ForProgramBlock fpb = (ForProgramBlock) pbParent;
			replaceProgramBlock( fpb.getChildBlocks(), pbOld, pbNew );	
		}
		else
			throw new DMLUnsupportedOperationException("Optimizer doesn't support "+pbParent.getClass().getName());
		
		//update repository
		_rtObjMap.replaceMapping(pbNew, n);	
	}
	/**
	 * 
	 * @param pbs
	 * @param pbOld
	 * @param pbNew
	 */
	public static void replaceProgramBlock(ArrayList<ProgramBlock> pbs, ProgramBlock pbOld, ProgramBlock pbNew)
	{
		int len = pbs.size();
		for( int i=0; i<len; i++ )
			if( pbs.get(i) == pbOld )
				pbs.set(i, pbNew);
	}
	
	/**
	 * 
	 * @param pb
	 * @param tree
	 * @param parforOnly
	 */
	public static void changeProgramBlock(ParForProgramBlock pb, OptTree tree, boolean parforOnly) 
	{
		/*
		if( parforOnly )
		{
			//changes only in top;level parfor
			OptNode parfor1 = tree.getRoot();
			OptNode parfor2 = null;
				
			PExecMode mode1 = (parfor1.getExecType()==ExecType.CP)? PExecMode.LOCAL : PExecMode.REMOTE_MR;
			int k1 = 0;
			
			// we need a better more generic version here.
		}
		else
		{
			// recursive invocation and parameter change
		}
		
		// if heuristic parforonly

		// internal use for create rtprograms
		// use interface to hops - lops generation

		// change the parameters of subprogramblocks  as well
		 
	*/
	}
		
	/**
	 * 
	 * @return
	 */
	public static RTObjectMapping getRTObjectMapping()
	{
		return _rtObjMap;
	}
	
	/**
	 * 
	 * @return
	 */
	public static HLObjectMapping getHLObjectMapping()
	{
		return _hlObjMap;
	}
	

	/**
	 * 
	 * @param pRoot
	 * @param hlNodeID
	 * @param newRtNode
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static OptNode exchangeTemporary(OptNode pRoot, long hlNodeID, OptNode newRtNode) 
		throws DMLRuntimeException 
	{
		OptNode hlNode = _hlObjMap.getOptNode(hlNodeID);
		if( hlNode.getNodeType() == NodeType.PARFOR )
		{
			ParForProgramBlock pb = (ParForProgramBlock) _hlObjMap.getMappedProg(hlNodeID)[1];
			OptNode rtNode = _rtObjMap.getOptNode(pb);
			
			//copy node internals (because it might be root node)
			_tmpChildOld = rtNode.createShallowClone();
			rtNode.setExecType(newRtNode.getExecType()); //TODO extend as required
		}
		else if (hlNode.getNodeType() == NodeType.HOP)
		{
			long pid1 = _hlObjMap.getMappedParentID(hlNode.getID()); //pbID
			ProgramBlock pb = (ProgramBlock) _hlObjMap.getMappedProg(pid1)[1];
			OptNode rtNode1 = _rtObjMap.getOptNode(pb);
			long pid2 = _rtObjMap.getMappedParentID(rtNode1.getID());
			OptNode rtNode2 = _rtObjMap.getOptNode(pid2);
			
			System.out.println("exchanging "+rtNode1.getNodeType()+" "+rtNode1.getID());
			_tmpParent = rtNode2;
			_tmpChildOld = rtNode1;		
			_tmpChildNew = newRtNode;
			System.out.println(_tmpParent.exchangeChild(_tmpChildOld, _tmpChildNew));
		}
		else
		{
			throw new DMLRuntimeException("Unexpected node type for plan node exchange.");
		}
		
		return pRoot;
	}
	
	/**
	 * 
	 * @param hlNodeID
	 * @throws DMLRuntimeException
	 */
	public static void revertTemporaryChange( long hlNodeID ) 
		throws DMLRuntimeException 
	{
		OptNode node = _hlObjMap.getOptNode(hlNodeID);
		
		if( node.getNodeType() == NodeType.PARFOR )
		{
			ParForProgramBlock pb = (ParForProgramBlock) _hlObjMap.getMappedProg(hlNodeID)[1];
			OptNode rtNode = _rtObjMap.getOptNode(pb);
			rtNode.setExecType(_tmpChildOld.getExecType()); 	
		}
		else if( node.getNodeType() == NodeType.HOP )
		{
			//revert change (overwrite tmp child)
			System.out.println( _tmpParent.exchangeChild(_tmpChildNew,_tmpChildOld) );	
		}
		else
		{
			throw new DMLRuntimeException("Unexpected node type for plan node exchange.");
		}
		
		//cleanup
		_tmpParent = null;
		_tmpChildOld = null;
	}

	/**
	 * 
	 * @param pRoot
	 * @param hlNodeID
	 * @param newRtNode
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static OptNode exchangePermanently(OptNode pRoot, long hlNodeID, OptNode newRtNode) 
		throws DMLRuntimeException 
	{
		OptNode hlNode = _hlObjMap.getOptNode(hlNodeID);
		if( hlNode.getNodeType() == NodeType.PARFOR )
		{
			ParForProgramBlock pb = (ParForProgramBlock) _hlObjMap.getMappedProg(hlNodeID)[1];
			OptNode rtNode = _rtObjMap.getOptNode(pb);
			
			//copy node internals (because it might be root node)
			//(no need for update mapping)
			rtNode.setExecType(newRtNode.getExecType()); //TODO extend as required
		}
		else if (hlNode.getNodeType() == NodeType.HOP)
		{
			long pid1 = _hlObjMap.getMappedParentID(hlNode.getID()); //pbID
			ProgramBlock pb = (ProgramBlock) _hlObjMap.getMappedProg(pid1)[1];
			OptNode rtNode1 = _rtObjMap.getOptNode(pb);
			long pid2 = _rtObjMap.getMappedParentID(rtNode1.getID());
			OptNode rtNode2 = _rtObjMap.getOptNode(pid2);
			
			System.out.println("exchanging "+rtNode1.getNodeType()+" "+rtNode1.getID());
			System.out.println(rtNode2.exchangeChild(rtNode1, newRtNode));
			
			//finally update mapping (all internal repositories)
			newRtNode.setID(rtNode1.getID());
			_rtObjMap.replaceMapping(pb, newRtNode);
		}
		else
		{
			throw new DMLRuntimeException("Unexpected node type for plan node exchange.");
		}
		
		return pRoot;
	}


	/**
	 * 
	 */
	public static void clear() 
	{
		if( _hlObjMap != null )
			_hlObjMap.clear();
		
		if( _rtObjMap != null )
			_rtObjMap.clear();
		
		_tmpParent = null;
		_tmpChildOld = null;
		_tmpChildNew = null;
	}

	/**
	 * Helper class for mapping nodes of the internal plan representation to statement blocks and 
	 * hops / function call statements of a given DML program.
	 *
	 */
	public static class HLObjectMapping
	{
		//internal repository for mapping rtprogs and optnodes
		private IDSequence _idSeq;
		private HashMap<Long, Object> _id_hlprog;
		private HashMap<Long, Object> _id_rtprog;
		private HashMap<Long, OptNode> _id_optnode;
	
		public HLObjectMapping( )
		{
			_idSeq = new IDSequence();
			_id_hlprog = new HashMap<Long, Object>();
			_id_rtprog = new HashMap<Long, Object>();
			_id_optnode = new HashMap<Long, OptNode>();
		}
		
		public long putHopMapping( Hops hops, OptNode n )
		{
			long id = _idSeq.getNextID();
			
			_id_hlprog.put(id, hops);
			_id_rtprog.put(id, null);
			_id_optnode.put(id, n);	
			
			n.setID(id);
			
			return id;
		}
		
		public long putProgMapping( StatementBlock sb, ProgramBlock pb, OptNode n )
		{
			long id = _idSeq.getNextID();
			
			_id_hlprog.put(id, sb);
			_id_rtprog.put(id, pb);
			_id_optnode.put(id, n);
			n.setID(id);
			
			return id;
		}
		
		/*public void replaceMapping( StatementBlock sb, OptNode n )
		{
			long id = n.getID();
			_id_rtprog.put(id, sb);
			_id_optnode.put(id, n);
		}*/
		
		public OptNode getOptNode( long id )
		{
			return _id_optnode.get(id);
		}
		
		public Hops getMappedHop( long id )
		{
			return (Hops)_id_hlprog.get( id );
		}
		
		public Object[] getMappedProg( long id )
		{
			Object[] ret = new Object[2];
			ret[0] = (StatementBlock)_id_hlprog.get( id );
			ret[1] = (ProgramBlock)_id_rtprog.get( id );
			
			return ret;
		}
		
		public long getMappedParentID( long id )
		{
			for( OptNode p : _id_optnode.values() )
				if( p.getChilds() != null )
					for( OptNode c2 : p.getChilds() )
						if( id == c2.getID() )
							return p.getID();
			return -1;
		}
		
		public void clear()
		{
			_id_hlprog.clear();
			_id_rtprog.clear();
			_id_optnode.clear();
		}
	}
	
	/**
	 * Helper class for mapping nodes of the internal plan representation to program blocks and 
	 * instructions of a given runtime program.
	 *
	 */
	public static class RTObjectMapping
	{
		//TODO shared super class for both mapping classes or central repository
		
		//internal repository for mapping rtprogs and optnodes
		private IDSequence _idSeq;
		private HashMap<Long, Object> _id_rtprog;
		private HashMap<Long, OptNode> _id_optnode;
	
		public RTObjectMapping( )
		{
			_idSeq = new IDSequence();
			_id_rtprog = new HashMap<Long, Object>();
			_id_optnode = new HashMap<Long, OptNode>();
		}
		
		public long putMapping( Instruction inst, OptNode n )
		{
			long id = _idSeq.getNextID();
			
			_id_rtprog.put(id, inst);
			_id_optnode.put(id, n);			
			n.setID(id);
			
			return id;
		}
		
		public long putMapping( ProgramBlock pb, OptNode n )
		{
			long id = _idSeq.getNextID();
			
			_id_rtprog.put(id, pb);
			_id_optnode.put(id, n);
			n.setID(id);
			
			return id;
		}
		
		public void replaceMapping( ProgramBlock pb, OptNode n )
		{
			long id = n.getID();
			_id_rtprog.put(id, pb);
			_id_optnode.put(id, n);
		}
		
		public Object getMappedObject( long id )
		{
			return _id_rtprog.get( id );
		}
		
		public long getMappedParentID( long id )
		{
			for( OptNode p : _id_optnode.values() )
				if( p.getChilds() != null )
					for( OptNode c2 : p.getChilds() )
						if( id == c2.getID() )
							return p.getID();
			return -1;
		}
		
		public OptNode getOptNode( long id )
		{
			return _id_optnode.get(id);
		}
		
		public OptNode getOptNode( Object prog )
		{
			for( Entry<Long,Object> e : _id_rtprog.entrySet() )
				if( e.getValue() == prog )
					return _id_optnode.get(e.getKey());
			return null;
		}
		
		public void clear()
		{
			_id_rtprog.clear();
			_id_optnode.clear();
		}
	}
}
