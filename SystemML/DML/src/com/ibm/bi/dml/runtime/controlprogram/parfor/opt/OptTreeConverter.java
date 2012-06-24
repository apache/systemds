package com.ibm.bi.dml.runtime.controlprogram.parfor.opt;

import java.util.ArrayList;
import java.util.HashMap;

import com.ibm.bi.dml.api.DMLScript;
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

/**
 * Converter for creating an internal plan representation for a given runtime program
 * and to modify/create the runtime program according to the optimized plan.
 * 
 * 
 */
public class OptTreeConverter 
{	
	private static ObjectMapping    _objMap;
	private static StatisticMapping _statMap;
	
	static
	{
		_objMap = new ObjectMapping();
		_statMap = new StatisticMapping();
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
		OptNode root = rCreateOptNode( pfpb, pfpb.getVariables(), true );		
		OptTree tree = new OptTree(ck, cm, root);
		
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
	private static OptNode rCreateOptNode( ProgramBlock pb, LocalVariableMap vars, boolean topLevel ) 
		throws DMLUnsupportedOperationException, DMLRuntimeException 
	{
		OptNode node = null;
		
		if( pb instanceof IfProgramBlock )
		{
			IfProgramBlock ipb = (IfProgramBlock) pb;
			node = new OptNode( NodeType.IF );
			_objMap.putMapping(ipb, node);
			node.setExecType(ExecType.CP);
			//process if condition
			OptNode ifn = new OptNode(NodeType.GENERIC);
			node.addChilds( createOptNodes( ipb.getPredicate(), vars ) );
			node.addChild( ifn );
			for( ProgramBlock lpb : ipb.getChildBlocksIfBody() )
				ifn.addChild( rCreateOptNode(lpb,vars,topLevel) );
			//process else condition
			if( ipb.getChildBlocksElseBody() != null )
			{
				OptNode efn = new OptNode(NodeType.GENERIC);
				node.addChild( efn );
				for( ProgramBlock lpb : ipb.getChildBlocksElseBody() )
					efn.addChild( rCreateOptNode(lpb,vars,topLevel) );
			}				
		}
		else if( pb instanceof WhileProgramBlock )
		{
			WhileProgramBlock wpb = (WhileProgramBlock) pb;
			node = new OptNode( NodeType.WHILE );
			_objMap.putMapping(wpb, node);
			node.setExecType(ExecType.CP);
			//process predicate instruction
			node.addChilds( createOptNodes( wpb.getPredicate(), vars ) );
			//process body
			for( ProgramBlock lpb : wpb.getChildBlocks() )
				node.addChild( rCreateOptNode(lpb,vars,topLevel) );
			
		}
		else if( pb instanceof ForProgramBlock && !(pb instanceof ParForProgramBlock) )
		{
			ForProgramBlock fpb = (ForProgramBlock) pb;
			node = new OptNode( NodeType.FOR );
			_objMap.putMapping(fpb, node);
			node.setExecType(ExecType.CP);
			
			node.addParam(ParamType.NUM_ITERATIONS, String.valueOf(CostEstimator.FACTOR_NUM_ITERATIONS));
			
			node.addChilds( createOptNodes( fpb.getFromInstructions(), vars ) );
			node.addChilds( createOptNodes( fpb.getToInstructions(), vars ) );
			node.addChilds( createOptNodes( fpb.getIncrementInstructions(), vars ) );
			
			//process body
			for( ProgramBlock lpb : fpb.getChildBlocks() )
				node.addChild( rCreateOptNode(lpb,vars,topLevel) );
		}
		else if( pb instanceof ParForProgramBlock )
		{
			ParForProgramBlock fpb = (ParForProgramBlock) pb;			
			node = new OptNode( NodeType.PARFOR );
			_objMap.putMapping(fpb, node);
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
				node.addChilds( createOptNodes( fpb.getFromInstructions(), vars ) );
				node.addChilds( createOptNodes( fpb.getToInstructions(), vars ) );
				node.addChilds( createOptNodes( fpb.getIncrementInstructions(), vars ) );
			}
			
			//process body
			for( ProgramBlock lpb : fpb.getChildBlocks() )
				node.addChild( rCreateOptNode(lpb,vars,false) );			
			
			//parameters, add required parameters
		}
		else //last level program block
		{
			node = new OptNode(NodeType.GENERIC);
			_objMap.putMapping(pb, node);
			node.addChilds( createOptNodes(pb.getInstructions(), vars) );
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
	private static ArrayList<OptNode> createOptNodes (ArrayList<Instruction> instset, LocalVariableMap vars) 
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		ArrayList<OptNode> tmp = new ArrayList<OptNode>(instset.size());
		for( Instruction inst : instset )
			tmp.add( createOptNode(inst,vars) );
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
	private static OptNode createOptNode( Instruction inst, LocalVariableMap vars ) 
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		OptNode node = new OptNode(NodeType.INST);
		String instStr = inst.toString();
		String opstr = instStr.split(Instruction.OPERAND_DELIM)[1];
		_objMap.putMapping(inst, node);
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
		
		_statMap.putStatistics(instStr, analyzeStatistics(inst, node, vars) );
		
		return node;
	}
	
	/**
	 * 
	 * @param inst
	 * @param on
	 * @param vars
	 * @return
	 * @throws DMLRuntimeException
	 */
	private static double[] analyzeStatistics(Instruction inst, OptNode on, LocalVariableMap vars) 
		throws DMLRuntimeException 
	{
		double[] ret = null;
		String instName = on.getInstructionName();
		
		if( PerfTestTool.isRegisteredInstruction(instName) )
		{	
			if( inst instanceof RandCPInstruction )
			{
				RandCPInstruction linst = (RandCPInstruction) inst;
				DataFormat df = (linst.sparsity>MatrixBlockDSM.SPARCITY_TURN_POINT) ? 
						                        DataFormat.DENSE : DataFormat.SPARSE; 
				ret = new double[]{linst.rows, linst.cols, -1, -1, linst.sparsity, df.ordinal()};
			}
			else if ( inst instanceof FunctionCallCPInstruction )
			{
				FunctionCallCPInstruction linst = (FunctionCallCPInstruction)inst;
				ArrayList<String> params = linst.getBoundInputParamNames();
				ret = StatisticMapping.DEFAULT_STATS.clone();
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
							ret[ 0 ] = mc1.numRows;
							ret[ 1 ] = mc1.numColumns;
							ret[ 4 ] = mc1.nonZero /( ret[0] * ret[1] ); //sparsity
							ret[ 5 ] = (ret[4] < MatrixBlockDSM.SPARCITY_TURN_POINT )? DataFormat.SPARSE.ordinal() : DataFormat.DENSE.ordinal(); 
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
				ret = StatisticMapping.DEFAULT_STATS.clone();
				
				if( linst.input1 != null && linst.input2 != null ) //binary
				{
					Data dat1 = vars.get( linst.input1.get_name() );
					Data dat2 = vars.get( linst.input2.get_name() );
					
					if( dat1 != null )
					{
						MatrixObjectNew mdat1 = (MatrixObjectNew) dat1;
						MatrixCharacteristics mc1 = ((MatrixFormatMetaData)mdat1.getMetaData()).getMatrixCharacteristics();
						ret[ 0 ] = mc1.numRows;
						ret[ 1 ] = mc1.numColumns;
						ret[ 4 ] = mc1.nonZero /( ret[0] * ret[1] ); //sparsity
						ret[ 5 ] = (ret[4] < MatrixBlockDSM.SPARCITY_TURN_POINT )? DataFormat.SPARSE.ordinal() : DataFormat.DENSE.ordinal(); 
					}
					if( dat2 != null )
					{
						MatrixObjectNew mdat2 = (MatrixObjectNew) dat2;
						MatrixCharacteristics mc2 = ((MatrixFormatMetaData)mdat2.getMetaData()).getMatrixCharacteristics();
						ret[ 2 ] = mc2.numRows;
						ret[ 3 ] = mc2.numColumns;
						ret[ 5 ] = (mc2.nonZero /( ret[2] * ret[3]) < MatrixBlockDSM.SPARCITY_TURN_POINT ) ? DataFormat.SPARSE.ordinal() : DataFormat.DENSE.ordinal(); 
					}
				}
				else //unary
				{
					Data dat1 = vars.get( linst.input1.get_name() );
					
					if( dat1 != null )
					{
						MatrixObjectNew mdat1 = (MatrixObjectNew) dat1;
						MatrixCharacteristics mc1 = ((MatrixFormatMetaData)mdat1.getMetaData()).getMatrixCharacteristics();
						ret[ 0 ] = mc1.numRows;
						ret[ 1 ] = mc1.numColumns;
						ret[ 4 ] = mc1.nonZero /( ret[0] * ret[1] ); //sparsity
						ret[ 5 ] = (ret[4] < MatrixBlockDSM.SPARCITY_TURN_POINT ) ? DataFormat.SPARSE.ordinal() : DataFormat.DENSE.ordinal(); 
					}					
				}
			}
		}
		
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
		ProgramBlock pbParent = (ProgramBlock)_objMap.getMappedObject( parent.getID() );
		
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
		_objMap.replaceMapping(pbNew, n);	
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
	 * @param node
	 * @return
	 */
	public static double[] getStatisticsMapping( OptNode node )
	{
		//get related instruction
		Instruction inst = (Instruction) _objMap.getMappedObject( node.getID() );
		String instStr = inst.toString();
			
		//get related statistics
		double[] ret = _statMap.getStatistics(instStr);
		if( ret == null ) //no stats available
			ret = StatisticMapping.DEFAULT_STATS;
		
		return ret; 
	}
	
	/**
	 * 
	 * @param statMap
	 */
	public static void setStatisticsMapping( StatisticMapping statMap ) 
	{
		_statMap = statMap;
		System.out.println("set stat mapping"); 
	}
	
	/**
	 * 
	 * @return
	 */
	public static StatisticMapping getStatisticsMapping()
	{
		return _statMap;
	}
	
	/**
	 * 
	 * @return
	 */
	public static ObjectMapping getObjectMapping()
	{
		return _objMap;
	}
	
	/**
	 * Helper class for mapping statistic information to program blocks and instructions of
	 * a given runtime program.
	 * 
	 */
	public static class StatisticMapping
	{
		public static final double     DEFAULT_DIMENSION  = 100;
		public static final double     DEFAULT_SPARSITY   = 1.0;		
		public static final DataFormat DEFAULT_DATAFORMAT = DataFormat.DENSE;
		public static final double[]   DEFAULT_STATS      = new double[]{DEFAULT_DIMENSION,DEFAULT_DIMENSION,
			                                                      DEFAULT_DIMENSION, DEFAULT_DIMENSION, DEFAULT_SPARSITY,
			                                                      DEFAULT_DATAFORMAT.ordinal()};		
		private HashMap<String,double[]> _instStats;
		
		public StatisticMapping()
		{
			_instStats = new HashMap<String, double[]>();
		}
		
		public void putStatistics( String inst, double[] stats )
		{
			if( stats != null )
				_instStats.put(inst, stats);
		}
		
		public double[] getStatistics( String inst )
		{
			return _instStats.get(inst);
		}
		
		public void clear()
		{
			_instStats.clear();
		}
	}
	
	/**
	 * Helper class for mapping nodes of the internal plan representation to program blocks and 
	 * instructions of a given runtime program.
	 *
	 */
	public static class ObjectMapping
	{
		//internal repository for mapping rtprogs and optnodes
		private IDSequence _idSeq;
		private HashMap<Long, Object> _id_rtprog;
		private HashMap<Long, OptNode> _id_optnode;
	
		public ObjectMapping( )
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
		
		public void clear()
		{
			_id_rtprog.clear();
			_id_optnode.clear();
		}
	}
}
