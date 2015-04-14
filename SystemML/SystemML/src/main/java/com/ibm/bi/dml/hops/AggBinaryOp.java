/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops;

import com.ibm.bi.dml.hops.rewrite.HopRewriteUtils;
import com.ibm.bi.dml.lops.Aggregate;
import com.ibm.bi.dml.lops.Binary;
import com.ibm.bi.dml.lops.DataPartition;
import com.ibm.bi.dml.lops.Group;
import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.LopsException;
import com.ibm.bi.dml.lops.MMCJ;
import com.ibm.bi.dml.lops.MMRJ;
import com.ibm.bi.dml.lops.MMTSJ;
import com.ibm.bi.dml.lops.MMTSJ.MMTSJType;
import com.ibm.bi.dml.lops.MapMult;
import com.ibm.bi.dml.lops.MapMultChain;
import com.ibm.bi.dml.lops.MapMultChain.ChainType;
import com.ibm.bi.dml.lops.PMMJ;
import com.ibm.bi.dml.lops.PartialAggregate.CorrectionLocationType;
import com.ibm.bi.dml.lops.Transform;
import com.ibm.bi.dml.lops.Transform.OperationTypes;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.PDataPartitionFormat;
import com.ibm.bi.dml.runtime.controlprogram.context.SparkExecutionContext;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.mapred.DistributedCacheInput;
import com.ibm.bi.dml.sql.sqllops.SQLCondition;
import com.ibm.bi.dml.sql.sqllops.SQLCondition.BOOLOP;
import com.ibm.bi.dml.sql.sqllops.SQLJoin;
import com.ibm.bi.dml.sql.sqllops.SQLLopProperties;
import com.ibm.bi.dml.sql.sqllops.SQLLopProperties.AGGREGATIONTYPE;
import com.ibm.bi.dml.sql.sqllops.SQLLopProperties.JOINTYPE;
import com.ibm.bi.dml.sql.sqllops.SQLLops;
import com.ibm.bi.dml.sql.sqllops.SQLLops.GENERATES;
import com.ibm.bi.dml.sql.sqllops.SQLSelectStatement;
import com.ibm.bi.dml.sql.sqllops.SQLTableReference;


/* Aggregate binary (cell operations): Sum (aij + bij)
 * 		Properties: 
 * 			Inner Symbol: *, -, +, ...
 * 			Outer Symbol: +, min, max, ...
 * 			2 Operands
 * 	
 * 		Semantic: generate indices, align, cross-operate, generate indices, align, aggregate
 */

public class AggBinaryOp extends Hop 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	public static final double MAPMULT_MEM_MULTIPLIER = 1.0;
	public static MMultMethod FORCED_MMULT_METHOD = null;
	
	private OpOp2 innerOp;
	private AggOp outerOp;

	public enum MMultMethod { 
		CPMM,     //cross-product matrix multiplication (mr)
		RMM,      //replication matrix multiplication (mr)
		MAPMM_L,  //map-side matrix-matrix multiplication using distributed cache (mr)
		MAPMM_R,  //map-side matrix-matrix multiplication using distributed cache (mr)
		MAPMM_CHAIN, //map-side matrix-matrix-matrix multiplication using distributed cache, for right input (mr)
		PMM,      //permutation matrix multiplication using distributed cache, for left input (mr/cp)
		TSMM,     //transpose-self matrix multiplication (mr/cp)
		CP        //in-memory matrix multiplication (cp)
	};
	
	//hints set by previous to operator selection
	private boolean _hasLeftPMInput = false; //left input is permutation matrix
	
	private AggBinaryOp() {
		//default constructor for clone
	}
	
	public AggBinaryOp(String l, DataType dt, ValueType vt, OpOp2 innOp,
			AggOp outOp, Hop in1, Hop in2) {
		super(Kind.AggBinaryOp, l, dt, vt);
		innerOp = innOp;
		outerOp = outOp;
		getInput().add(0, in1);
		getInput().add(1, in2);
		in1.getParent().add(this);
		in2.getParent().add(this);
		
		//compute unknown dims and nnz
		refreshSizeInformation();
	}
	
	public void setHasLeftPMInput(boolean flag) {
		_hasLeftPMInput = flag;
	}
	
	/**
	 * NOTE: overestimated mem in case of transpose-identity matmult, but 3/2 at worst
	 *       and existing mem estimate advantageous in terms of consistency hops/lops,
	 *       and some special cases internally materialize the transpose for better cache locality  
	 */
	@Override
	public Lop constructLops() 
		throws HopsException, LopsException 
	{
		//return already created lops
		if( getLops() != null )
			return getLops();
	
		//construct matrix mult lops (currently only supported aggbinary)
		if ( isMatrixMultiply() ) 
		{
			Hop input1 = getInput().get(0);
			Hop input2 = getInput().get(1);
			
			//matrix mult operation selection part 1 (CP vs MR)
			ExecType et = optFindExecType();
			MMTSJType mmtsj = checkTransposeSelf();
		
			if ( et == ExecType.CP ) 
			{
				if( mmtsj != MMTSJType.NONE ) { //CP TSMM
					constructLopsCP_TSMM( mmtsj );
				}
				else if( _hasLeftPMInput && input1.getDim2()==1 
						&& input2.getDim1()!=1 ) //CP PMM
				{
					constructLopsCP_PMM();
				}
				else { //CP MM
					constructLopsCP_MM();
				}
					
			}
			else if ( et == ExecType.SPARK ) 
			{
				MMultMethod method = optFindMMultMethodSpark ( 
						input1.getDim1(), input1.getDim2(), input1.getRowsInBlock(), input1.getColsInBlock(),    
						input2.getDim1(), input2.getDim2(), input2.getRowsInBlock(), input2.getColsInBlock(), mmtsj);
			
				switch( method )
				{
					case TSMM:
						Hop input = getInput().get((mmtsj==MMTSJType.LEFT)?1:0);
						MMTSJ tsmm = new MMTSJ(input.constructLops(), getDataType(), getValueType(), et, mmtsj);
						setOutputDimensions(tsmm);
						setLineNumbers(tsmm);
						setLops(tsmm);
						
						break;
						
					case MAPMM_L:
					case MAPMM_R:
						// If number of columns is smaller than block size then explicit aggregation is not required.
						// i.e., entire matrix multiplication can be performed in the mappers.
						boolean needAgg = requiresAggregation(method); 
						_outputEmptyBlocks = !OptimizerUtils.allowsToFilterEmptyBlockOutputs(this); 
						
						//core matrix mult
						MapMult mapmult = new MapMult( getInput().get(0).constructLops(), getInput().get(1).constructLops(), 
								                getDataType(), getValueType(), (method==MMultMethod.MAPMM_R), false, 
								                _outputEmptyBlocks, needAgg);
						setOutputDimensions(mapmult);
						setLineNumbers(mapmult);
						setLops(mapmult);
						
						break;
						
					case CPMM:	
						// SPARK_INTEGRATION: MMCJ
						// TODO: Implement tsmm, pmm, and left transpose rewrite of CP_MM
						Lop matmultCP = new Binary(getInput().get(0).constructLops(),getInput().get(1).constructLops(), 
													 Binary.OperationTypes.MATMULT, getDataType(), getValueType(), optFindExecType());
						matmultCP.getOutputParameters().setDimensions(getDim1(), getDim2(), getRowsInBlock(), getColsInBlock(), getNnz());
						setLineNumbers( matmultCP );
						setLops(matmultCP);
						
						break;
				}
			}
			else if ( et == ExecType.MR ) 
			{
				//matrix mult operation selection part 2 (MR type)
				ChainType chain = checkMapMultChain();
				MMultMethod method = optFindMMultMethodMR ( 
							input1.getDim1(), input1.getDim2(), input1.getRowsInBlock(), input1.getColsInBlock(),    
							input2.getDim1(), input2.getDim2(), input2.getRowsInBlock(), input2.getColsInBlock(),
							mmtsj, chain, _hasLeftPMInput);
			
				//dispatch lops construction
				switch( method ) {
					case MAPMM_L:
					case MAPMM_R: 	
						constructLopsMR_MapMM(method); break;
					
					case MAPMM_CHAIN:	
						constructLopsMR_MapMMChain( chain ); break;		
					
					case CPMM:
						constructLopsMR_CPMM(); break;					
					
					case RMM:			
						constructLopsMR_RMM(); break;
						
					case TSMM:
						constructLopsMR_TSMM( mmtsj ); break;
					
					case PMM:
						constructLopsMR_PMM(); break;
						
					default:
						throw new HopsException(this.printErrorLocation() + "Invalid Matrix Mult Method (" + method + ") while constructing lops.");
				}
			}
		} 
		else
			throw new HopsException(this.printErrorLocation() + "Invalid operation in AggBinary Hop, aggBin(" + innerOp + "," + outerOp + ") while constructing lops.");
		
		return getLops();
	}

	@Override
	public String getOpString() {
		//ba - binary aggregate, for consistency with runtime 
		String s = "ba(" + 
				HopsAgg2String.get(outerOp) + 
				HopsOpOp2String.get(innerOp)+")";
		return s;
	}

	public void printMe() throws HopsException {
		if (LOG.isDebugEnabled()){
			if (getVisited() != VisitStatus.DONE) {
				super.printMe();
				LOG.debug("  InnerOperation: " + innerOp);
				LOG.debug("  OuterOperation: " + outerOp);
				for (Hop h : getInput()) {
					h.printMe();
				}
				;
			}
			setVisited(VisitStatus.DONE);
		}
	}

	@Override
	protected double computeOutputMemEstimate( long dim1, long dim2, long nnz )
	{		
		//NOTES:  
		// * The estimate for transpose-self is the same as for normal matrix multiplications
		//   because (1) this decouples the decision of TSMM over default MM and (2) some cases
		//   of TSMM internally materialize the transpose for efficiency.
		// * All matrix multiplications internally use dense output representations for efficiency.
		//   This is reflected in our conservative memory estimate. However, we additionally need 
		//   to account for potential final dense/sparse transformations via processing mem estimates.
		double sparsity = 1.0;
		/*
		if( isMatrixMultiply() ) {	
			if( nnz < 0 ){
				Hops input1 = getInput().get(0);
				Hops input2 = getInput().get(1);
				if( input1.dimsKnown() && input2.dimsKnown() )
				{
					double sp1 = (input1.getNnz()>0) ? OptimizerUtils.getSparsity(input1.getDim1(), input1.getDim2(), input1.getNnz()) : 1.0;
					double sp2 = (input2.getNnz()>0) ? OptimizerUtils.getSparsity(input2.getDim1(), input2.getDim2(), input2.getNnz()) : 1.0;
					sparsity = OptimizerUtils.getMatMultSparsity(sp1, sp2, input1.getDim1(), input1.getDim2(), input2.getDim2(), true);	
				}
			}
			else //sparsity known (e.g., inferred from worst case estimates)
				sparsity = OptimizerUtils.getSparsity(dim1, dim2, nnz);
		}
		*/
		//currently always estimated as dense in order to account for dense intermediate without unnecessary overestimation 
		double ret = OptimizerUtils.estimateSizeExactSparsity(dim1, dim2, sparsity);
		
		return ret;
	}
	
	@Override
	protected double computeIntermediateMemEstimate( long dim1, long dim2, long nnz )
	{
		double ret = 0;
		
		//account for potential final dense-sparse transformation (worst-case sparse representation)
		if( dim2 >= 2 ) //vectors always dense
			ret = OptimizerUtils.estimateSizeExactSparsity(dim1, dim2, MatrixBlock.SPARSITY_TURN_POINT);
		
		return ret;
	}
	
	@Override
	protected long[] inferOutputCharacteristics( MemoTable memo )
	{
		long[] ret = null;
	
		MatrixCharacteristics[] mc = memo.getAllInputStats(getInput());
		if( mc[0].rowsKnown() && mc[1].colsKnown() ) {
			ret = new long[3];
			ret[0] = mc[0].getRows();
			ret[1] = mc[1].getCols();
			double sp1 = (mc[0].getNonZeros()>0) ? OptimizerUtils.getSparsity(mc[0].getRows(), mc[0].getCols(), mc[0].getNonZeros()) : 1.0; 
			double sp2 = (mc[1].getNonZeros()>0) ? OptimizerUtils.getSparsity(mc[1].getRows(), mc[1].getCols(), mc[1].getNonZeros()) : 1.0; 			
			ret[2] = (long) ( ret[0] * ret[1] * OptimizerUtils.getMatMultSparsity(sp1, sp2, ret[0], mc[0].getCols(), ret[1], true));
		}
		
		return ret;
	}
	

	public boolean isMatrixMultiply() {
		return ( this.innerOp == OpOp2.MULT && this.outerOp == AggOp.SUM );			
	}
	
	private boolean isOuterProduct() {
		if ( getInput().get(0).isVector() && getInput().get(1).isVector() ) {
			if ( getInput().get(0).getDim1() == 1 && getInput().get(0).getDim1() > 1
					&& getInput().get(1).getDim1() > 1 && getInput().get(1).getDim2() == 1 )
				return true;
			else
				return false;
		}
		return false;
	}
	
	@Override
	public boolean allowsAllExecTypes()
	{
		return true;
	}
	
	@Override
	protected ExecType optFindExecType() {
		
		checkAndSetForcedPlatform();

		if( _etypeForced != null ) 			
		{
			_etype = _etypeForced;
		}
		else 
		{
			if ( OptimizerUtils.isMemoryBasedOptLevel() ) 
			{
				_etype = findExecTypeByMemEstimate();
			}
			// choose CP if the dimensions of both inputs are below Hops.CPThreshold 
			// OR if it is vector-vector inner product
			else if ( (getInput().get(0).areDimsBelowThreshold() && getInput().get(1).areDimsBelowThreshold())
						|| (getInput().get(0).isVector() && getInput().get(1).isVector() && !isOuterProduct()) )
			{
				_etype = ExecType.CP;
			}
			else
			{
				_etype = ExecType.MR;
			}
			
			//check for valid CP dimensions and matrix size
			checkAndSetInvalidCPDimsAndSize();
			
			//mark for recompile (forever)
			if(    OptimizerUtils.ALLOW_DYN_RECOMPILATION && !dimsKnown(true) 
				&& (_etype==ExecType.MR || _etype==ExecType.SPARK) )
			{
				setRequiresRecompile();			
			}
		}
		
		return _etype;
	}
	
	/**
	 * TSMM: Determine if XtX pattern applies for this aggbinary and if yes
	 * which type. 
	 * 
	 * @return
	 */
	public MMTSJType checkTransposeSelf()
	{
		MMTSJType ret = MMTSJType.NONE;
		
		Hop in1 = getInput().get(0);
		Hop in2 = getInput().get(1);
		
		if(    in1 instanceof ReorgOp 
			&& ((ReorgOp)in1).getOp() == ReOrgOp.TRANSPOSE 
			&& in1.getInput().get(0) == in2 )
		{
			ret = MMTSJType.LEFT;
		}
		
		if(    in2 instanceof ReorgOp 
			&& ((ReorgOp)in2).getOp() == ReOrgOp.TRANSPOSE 
			&& in2.getInput().get(0) == in1 )
		{
			ret = MMTSJType.RIGHT;
		}
		
		return ret;
	}

	/**
	 * MapMultChain: Determine if XtwXv/XtXv pattern applies for this aggbinary 
	 * and if yes which type. 
	 * 
	 * @return
	 */
	public ChainType checkMapMultChain()
	{
		ChainType chainType = ChainType.NONE;
		
		Hop in1 = getInput().get(0);
		Hop in2 = getInput().get(1);
		
		//check for transpose left input (both chain types)
		if( in1 instanceof ReorgOp && ((ReorgOp)in1).getOp() == ReOrgOp.TRANSPOSE )
		{
			Hop X = in1.getInput().get(0);
				
			//check mapmultchain patterns
			//t(X)%*%(w*(X%*%v))
			if( in2 instanceof BinaryOp )
			{
				Hop in3b = in2.getInput().get(1);
				if( in3b instanceof AggBinaryOp )
				{
					Hop in4 = in3b.getInput().get(0);
					if( X == in4 ) //common input
						chainType = ChainType.XtwXv;
				}
			}
			//t(X)%*%(X%*%v)
			else if( in2 instanceof AggBinaryOp )
			{
				Hop in3 = in2.getInput().get(0);
				if( X == in3 ) //common input
					chainType = ChainType.XtXv;
			}
		}
		
		return chainType;
	}
	
	/**
	 * 
	 * @throws HopsException
	 * @throws LopsException
	 */
	private void constructLopsCP_MM() 
		throws HopsException, LopsException
	{
		Lop matmultCP = null;
		if( isLeftTransposeRewriteApplicable(true, false) )
			matmultCP = constructCPLopWithLeftTransposeRewrite();
		else
			matmultCP = new Binary(getInput().get(0).constructLops(),getInput().get(1).constructLops(), 
									 Binary.OperationTypes.MATMULT, getDataType(), getValueType(), ExecType.CP);
		
		matmultCP.getOutputParameters().setDimensions(getDim1(), getDim2(), getRowsInBlock(), getColsInBlock(), getNnz());
		setLineNumbers( matmultCP );
		setLops(matmultCP);
	}
	
	/**
	 * 
	 * @param mmtsj
	 * @throws HopsException
	 * @throws LopsException
	 */
	private void constructLopsCP_TSMM( MMTSJType mmtsj ) 
		throws HopsException, LopsException
	{
		Lop matmultCP = new MMTSJ(getInput().get((mmtsj==MMTSJType.LEFT)?1:0).constructLops(),
				                 getDataType(), getValueType(), ExecType.CP, mmtsj);
	
		matmultCP.getOutputParameters().setDimensions(getDim1(), getDim2(), getRowsInBlock(), getColsInBlock(), getNnz());
		setLineNumbers( matmultCP );
		setLops(matmultCP);
	}
	
	/**
	 * NOTE: exists for consistency since removeEmtpy might be scheduled to MR
	 * but matrix mult on small output might be scheduled to CP. Hence, we 
	 * need to handle directly passed selection vectors in CP as well.
	 * 
	 * @throws HopsException
	 * @throws LopsException
	 */
	private void constructLopsCP_PMM() 
		throws HopsException, LopsException
	{
		Hop pmInput = getInput().get(0);
		Hop rightInput = getInput().get(1);
		
		Hop nrow = HopRewriteUtils.createValueHop(pmInput, true); //NROW
		HopRewriteUtils.setOutputBlocksizes(nrow, 0, 0);
		nrow.setForcedExecType(ExecType.CP);
		HopRewriteUtils.copyLineNumbers(this, nrow);
		Lop lnrow = nrow.constructLops();
		
		PMMJ pmm = new PMMJ(pmInput.constructLops(), rightInput.constructLops(), lnrow, getDataType(), getValueType(), false, false, ExecType.CP);
		pmm.getOutputParameters().setDimensions(getDim1(), getDim2(), getRowsInBlock(), getColsInBlock(), getNnz());
		setLineNumbers(pmm);
		
		setLops(pmm);
		
		HopRewriteUtils.removeChildReference(pmInput, nrow);
	}
	
	/**
	 * 
	 * @param method
	 * @throws HopsException
	 * @throws LopsException
	 */
	private void constructLopsMR_MapMM(MMultMethod method) 
		throws HopsException, LopsException
	{
		if( method == MMultMethod.MAPMM_R && isLeftTransposeRewriteApplicable(false, true) )
		{
			setLops( constructMapMultMRLopWithLeftTransposeRewrite() );
		}
		else //GENERAL CASE
		{	
			// If number of columns is smaller than block size then explicit aggregation is not required.
			// i.e., entire matrix multiplication can be performed in the mappers.
			boolean needAgg = requiresAggregation(method); 
			boolean needPart = requiresPartitioning(method, false);
			_outputEmptyBlocks = !OptimizerUtils.allowsToFilterEmptyBlockOutputs(this); 
			
			//pre partitioning 
			Lop leftInput = getInput().get(0).constructLops(); 
			Lop rightInput = getInput().get(1).constructLops();
			if( needPart ) {
				if( (method==MMultMethod.MAPMM_L) ) //left in distributed cache
				{
					Hop input = getInput().get(0);
					ExecType etPart = (OptimizerUtils.estimateSizeExactSparsity(input.getDim1(), input.getDim2(), OptimizerUtils.getSparsity(input.getDim1(), input.getDim2(), input.getNnz())) 
					          < OptimizerUtils.getLocalMemBudget()) ? ExecType.CP : ExecType.MR; //operator selection
					leftInput = new DataPartition(input.constructLops(), DataType.MATRIX, ValueType.DOUBLE, etPart, PDataPartitionFormat.COLUMN_BLOCK_WISE_N);
					leftInput.getOutputParameters().setDimensions(input.getDim1(), input.getDim2(), getRowsInBlock(), getColsInBlock(), input.getNnz());
					setLineNumbers(leftInput);
				}
				else //right side in distributed cache
				{
					Hop input = getInput().get(1);
					ExecType etPart = (OptimizerUtils.estimateSizeExactSparsity(input.getDim1(), input.getDim2(), OptimizerUtils.getSparsity(input.getDim1(), input.getDim2(), input.getNnz())) 
					          < OptimizerUtils.getLocalMemBudget()) ? ExecType.CP : ExecType.MR; //operator selection
					rightInput = new DataPartition(input.constructLops(), DataType.MATRIX, ValueType.DOUBLE, etPart, PDataPartitionFormat.ROW_BLOCK_WISE_N);
					rightInput.getOutputParameters().setDimensions(input.getDim1(), input.getDim2(), getRowsInBlock(), getColsInBlock(), input.getNnz());
					setLineNumbers(rightInput);
				}
			}					
			
			//core matrix mult
			MapMult mapmult = new MapMult( leftInput, rightInput, getDataType(), getValueType(), 
					                (method==MMultMethod.MAPMM_R), needPart, _outputEmptyBlocks);
			mapmult.getOutputParameters().setDimensions(getDim1(), getDim2(), getRowsInBlock(), getColsInBlock(), getNnz());
			setLineNumbers(mapmult);
			
			//post aggregation
			if (needAgg) {
				Group grp = new Group(mapmult, Group.OperationTypes.Sort, getDataType(), getValueType());
				Aggregate agg1 = new Aggregate(grp, HopsAgg2Lops.get(outerOp), getDataType(), getValueType(), ExecType.MR);
				
				grp.getOutputParameters().setDimensions(getDim1(), getDim2(), getRowsInBlock(), getColsInBlock(), getNnz());
				agg1.getOutputParameters().setDimensions(getDim1(), getDim2(), getRowsInBlock(), getColsInBlock(), getNnz());
				setLineNumbers(agg1);
				
				// aggregation uses kahanSum but the inputs do not have correction values
				agg1.setupCorrectionLocation(CorrectionLocationType.NONE);  
				
				setLops(agg1);
			}
			else {
				setLops(mapmult);
			}
		}	
	} 
	
	/**
	 * 
	 * @param chainType
	 * @throws HopsException
	 * @throws LopsException
	 */
	private void constructLopsMR_MapMMChain( ChainType chainType ) 
		throws HopsException, LopsException
	{
		Lop mapmult = null; 
		
		if( chainType == ChainType.XtXv )
		{
			//v never needs partitioning because always single block
			Hop hX = getInput().get(0).getInput().get(0);
			Hop hv = getInput().get(1).getInput().get(1);
			
			//core matrix mult
			mapmult = new MapMultChain( hX.constructLops(), hv.constructLops(), getDataType(), getValueType());
			mapmult.getOutputParameters().setDimensions(getDim1(), getDim2(), getRowsInBlock(), getColsInBlock(), getNnz());
			setLineNumbers(mapmult);
		}
		else //if( chainType == ChainType.XtwXv )
		{
			//v never needs partitioning because always single block
			Hop hX = getInput().get(0).getInput().get(0);
			Hop hw = getInput().get(1).getInput().get(0);
			Hop hv = getInput().get(1).getInput().get(1).getInput().get(1);
			
			double mestW = OptimizerUtils.estimateSize(hw.getDim1(), hw.getDim2());
			boolean needPart = !hw.dimsKnown() || hw.getDim1() * hw.getDim2() > DistributedCacheInput.PARTITION_SIZE;
			Lop X = hX.constructLops(), v = hv.constructLops(), w = null;
			if( needPart ){ //requires partitioning
				w = new DataPartition(hw.constructLops(), DataType.MATRIX, ValueType.DOUBLE, (mestW>OptimizerUtils.getLocalMemBudget())?ExecType.MR:ExecType.CP, PDataPartitionFormat.ROW_BLOCK_WISE_N);
				w.getOutputParameters().setDimensions(hw.getDim1(), hw.getDim2(), getRowsInBlock(), getColsInBlock(), hw.getNnz());
				setLineNumbers(w);	
			}
			else
				w = hw.constructLops();
			
			//core matrix mult
			mapmult = new MapMultChain( X, v, w, getDataType(), getValueType());
			mapmult.getOutputParameters().setDimensions(getDim1(), getDim2(), getRowsInBlock(), getColsInBlock(), getNnz());
			setLineNumbers(mapmult);
		}
		
		//post aggregation
		Group grp = new Group(mapmult, Group.OperationTypes.Sort, getDataType(), getValueType());
		grp.getOutputParameters().setDimensions(getDim1(), getDim2(), getRowsInBlock(), getColsInBlock(), getNnz());
		Aggregate agg1 = new Aggregate(grp, HopsAgg2Lops.get(outerOp), getDataType(), getValueType(), ExecType.MR);
		agg1.getOutputParameters().setDimensions(getDim1(), getDim2(), getRowsInBlock(), getColsInBlock(), getNnz());
		agg1.setupCorrectionLocation(CorrectionLocationType.NONE); // aggregation uses kahanSum 
		setLineNumbers(agg1);
		 
		setLops(agg1);
	} 
	
	/**
	 * 
	 * @throws HopsException
	 * @throws LopsException
	 */
	private void constructLopsMR_CPMM() 
		throws HopsException, LopsException
	{
		if( isLeftTransposeRewriteApplicable(false, false) )
		{
			setLops( constructMMCJMRLopWithLeftTransposeRewrite() );
		} 
		else //general case
		{
			MMCJ mmcj = new MMCJ(getInput().get(0).constructLops(), getInput().get(1).constructLops(), 
					             getDataType(), getValueType());
			mmcj.getOutputParameters().setDimensions(getDim1(), getDim2(), getRowsInBlock(), getColsInBlock(), getNnz());
			setLineNumbers(mmcj);
			
			Group grp = new Group(mmcj, Group.OperationTypes.Sort, getDataType(), getValueType());
			grp.getOutputParameters().setDimensions(getDim1(), getDim2(), getRowsInBlock(), getColsInBlock(), getNnz());
			setLineNumbers(grp);
			
			Aggregate agg1 = new Aggregate(grp, HopsAgg2Lops.get(outerOp), getDataType(), getValueType(), ExecType.MR);
			agg1.getOutputParameters().setDimensions(getDim1(), getDim2(), getRowsInBlock(), getColsInBlock(), getNnz());
			setLineNumbers(agg1);
			
			// aggregation uses kahanSum but the inputs do not have correction values
			agg1.setupCorrectionLocation(CorrectionLocationType.NONE);  
			
			setLops(agg1);
		}
	} 
	
	/**
	 * 
	 * @throws HopsException
	 * @throws LopsException
	 */
	private void constructLopsMR_RMM() 
		throws HopsException, LopsException
	{
		MMRJ rmm = new MMRJ(getInput().get(0).constructLops(), getInput().get(1).constructLops(), 
				            getDataType(), getValueType());
		rmm.getOutputParameters().setDimensions(getDim1(), getDim2(),getRowsInBlock(), getColsInBlock(), getNnz());
		setLineNumbers(rmm);
		
		setLops(rmm);
	} 
	
	/**
	 * 
	 * @param mmtsj
	 * @throws HopsException
	 * @throws LopsException
	 */
	private void constructLopsMR_TSMM(MMTSJType mmtsj) 
		throws HopsException, LopsException
	{
		Hop input = getInput().get((mmtsj==MMTSJType.LEFT)?1:0);
		
		MMTSJ tsmm = new MMTSJ(input.constructLops(), getDataType(), getValueType(), ExecType.MR, mmtsj);
		tsmm.getOutputParameters().setDimensions(getDim1(), getDim2(), getRowsInBlock(), getColsInBlock(), getNnz());
		setLineNumbers(tsmm);
		
		Aggregate agg1 = new Aggregate(tsmm, HopsAgg2Lops.get(outerOp), getDataType(), getValueType(), ExecType.MR);
		agg1.getOutputParameters().setDimensions(getDim1(), getDim2(), getRowsInBlock(), getColsInBlock(), getNnz());
		agg1.setupCorrectionLocation(CorrectionLocationType.NONE); // aggregation uses kahanSum but the inputs do not have correction values
		setLineNumbers(agg1);
		
		setLops(agg1);
	} 
	
	/**
	 * 
	 * @throws HopsException
	 * @throws LopsException
	 */
	private void constructLopsMR_PMM() 
		throws HopsException, LopsException
	{
		//PMM has two potential modes (a) w/ full permutation matrix input, and 
		//(b) w/ already condensed input vector of target row positions.
		
		Hop pmInput = getInput().get(0);
		Hop rightInput = getInput().get(1);
		long brlen = pmInput.getRowsInBlock();
		long bclen = pmInput.getColsInBlock();
		
		//a) full permutation matrix input (potentially without empty block materialized)
		Lop lpmInput = pmInput.constructLops();
		if( pmInput.getDim2() != 1 ) //not a vector
		{
			//compute condensed permutation matrix vector input			
			//v = rowMaxIndex(t(pm)) * rowMax(t(pm)) 
			ReorgOp transpose = new ReorgOp( "tmp1", DataType.MATRIX, ValueType.DOUBLE, ReOrgOp.TRANSPOSE, pmInput );
			HopRewriteUtils.setOutputBlocksizes(transpose, brlen, bclen);
			transpose.refreshSizeInformation();
			transpose.setForcedExecType(ExecType.MR);
			HopRewriteUtils.copyLineNumbers(this, transpose);	
			
			AggUnaryOp agg1 = new AggUnaryOp("tmp2a", DataType.MATRIX, ValueType.DOUBLE, AggOp.MAXINDEX, Direction.Row, transpose);
			HopRewriteUtils.setOutputBlocksizes(agg1, brlen, bclen);
			agg1.refreshSizeInformation();
			agg1.setForcedExecType(ExecType.MR);
			HopRewriteUtils.copyLineNumbers(this, agg1);
			
			AggUnaryOp agg2 = new AggUnaryOp("tmp2b", DataType.MATRIX, ValueType.DOUBLE, AggOp.MAX, Direction.Row, transpose);
			HopRewriteUtils.setOutputBlocksizes(agg2, brlen, bclen);
			agg2.refreshSizeInformation();
			agg2.setForcedExecType(ExecType.MR);
			HopRewriteUtils.copyLineNumbers(this, agg2);
			
			BinaryOp mult = new BinaryOp("tmp3", DataType.MATRIX, ValueType.DOUBLE, OpOp2.MULT, agg1, agg2);
			HopRewriteUtils.setOutputBlocksizes(mult, brlen, bclen); 
			mult.refreshSizeInformation();
			mult.setForcedExecType(ExecType.MR);
			//mult.computeMemEstimate(memo); //select exec type
			HopRewriteUtils.copyLineNumbers(this, mult);
			
			lpmInput = mult.constructLops();
			
			HopRewriteUtils.removeChildReference(pmInput, transpose);
		}
		
		//b) condensed permutation matrix vector input (target rows)
		Hop nrow = HopRewriteUtils.createValueHop(pmInput, true); //NROW
		HopRewriteUtils.setOutputBlocksizes(nrow, 0, 0);
		nrow.setForcedExecType(ExecType.CP);
		HopRewriteUtils.copyLineNumbers(this, nrow);
		Lop lnrow = nrow.constructLops();
		
		boolean needPart = !pmInput.dimsKnown() || pmInput.getDim1() > DistributedCacheInput.PARTITION_SIZE;
		double mestPM = OptimizerUtils.estimateSize(pmInput.getDim1(), 1);
		if( needPart ){ //requires partitioning
			lpmInput = new DataPartition(lpmInput, DataType.MATRIX, ValueType.DOUBLE, (mestPM>OptimizerUtils.getLocalMemBudget())?ExecType.MR:ExecType.CP, PDataPartitionFormat.ROW_BLOCK_WISE_N);
			lpmInput.getOutputParameters().setDimensions(pmInput.getDim1(), 1, getRowsInBlock(), getColsInBlock(), pmInput.getDim1());
			setLineNumbers(lpmInput);	
		}
		
		_outputEmptyBlocks = !OptimizerUtils.allowsToFilterEmptyBlockOutputs(this); 
		PMMJ pmm = new PMMJ(lpmInput, rightInput.constructLops(), lnrow, getDataType(), getValueType(), needPart, _outputEmptyBlocks, ExecType.MR);
		pmm.getOutputParameters().setDimensions(getDim1(), getDim2(), getRowsInBlock(), getColsInBlock(), getNnz());
		setLineNumbers(pmm);
		
		Aggregate aggregate = new Aggregate(pmm, HopsAgg2Lops.get(outerOp), getDataType(), getValueType(), ExecType.MR);
		aggregate.getOutputParameters().setDimensions(getDim1(), getDim2(), getRowsInBlock(), getColsInBlock(), getNnz());
		aggregate.setupCorrectionLocation(CorrectionLocationType.NONE); // aggregation uses kahanSum but the inputs do not have correction values
		setLineNumbers(aggregate);
		
		setLops(aggregate);
		
		HopRewriteUtils.removeChildReference(pmInput, nrow);		
	} 
	
	
	
	/**
	 * Determines if the rewrite t(X)%*%Y -> t(t(Y)%*%X) is applicable
	 * and cost effective. Whenever X is a wide matrix and Y is a vector
	 * this has huge impact, because the transpose of X would dominate
	 * the entire operation costs.
	 * 
	 * @return
	 */
	private boolean isLeftTransposeRewriteApplicable(boolean CP, boolean checkMemMR)
	{
		boolean ret = false;
		Hop h1 = getInput().get(0);
		Hop h2 = getInput().get(1);
		
		//check for known dimensions and cost for t(X) vs t(v) + t(tvX)
		//(for both CP/MR, we explicitly check that new transposes fit in memory,
		//even a ba in CP does not imply that both transposes can be executed in CP)
		if( CP ) //in-memory ba 
		{
			if( h1 instanceof ReorgOp && ((ReorgOp)h1).getOp()==ReOrgOp.TRANSPOSE )
			{
				long m = h1.getDim1();
				long cd = h1.getDim2();
				long n = h2.getDim2();
				
				//check for known dimensions (necessary condition for subsequent checks)
				ret = (m>0 && cd>0 && n>0); 
				
				//check operation memory with changed transpose (this is important if we have 
				//e.g., t(X) %*% v, where X is sparse and tX fits in memory but X does not
				double memX = h1.getInput().get(0).getOutputMemEstimate();
				double memtv = OptimizerUtils.estimateSizeExactSparsity(n, cd, 1.0);
				double memtXv = OptimizerUtils.estimateSizeExactSparsity(n, m, 1.0);
				double newMemEstimate = memtv + memX + memtXv;
				ret &= ( newMemEstimate < OptimizerUtils.getLocalMemBudget() );
				
				//check for cost benefit of t(X) vs t(v) + t(tvX) and memory of additional transpose ops
				ret &= ( m*cd > (cd*n + m*n) &&
					2 * OptimizerUtils.estimateSizeExactSparsity(cd, n, 1.0) < OptimizerUtils.getLocalMemBudget() &&
					2 * OptimizerUtils.estimateSizeExactSparsity(m, n, 1.0) < OptimizerUtils.getLocalMemBudget() ); 
				
				//update operation memory estimate (e.g., for parfor optimizer)
				if( ret )
					_memEstimate = newMemEstimate;
			}
		}
		else //MR
		{
			if( h1 instanceof ReorgOp && ((ReorgOp)h1).getOp()==ReOrgOp.TRANSPOSE )
			{
				long m = h1.getDim1();
				long cd = h1.getDim2();
				long n = h2.getDim2();
				
				
				//note: output size constraint for mapmult already checked by optfindmmultmethod
				if( m>0 && cd>0 && n>0 && (m*cd > (cd*n + m*n)) &&
					2 * OptimizerUtils.estimateSizeExactSparsity(cd, n, 1.0) <  OptimizerUtils.getLocalMemBudget() &&
					2 * OptimizerUtils.estimateSizeExactSparsity(m, n, 1.0) <  OptimizerUtils.getLocalMemBudget() &&
					(!checkMemMR || OptimizerUtils.estimateSizeExactSparsity(cd, n, 1.0) < OptimizerUtils.getRemoteMemBudgetMap(true)) ) 
				{
					ret = true;
				}
			}
		}
		
		return ret;
	}
	
	/**
	 * 
	 * @return
	 * @throws HopsException
	 * @throws LopsException
	 */
	private Lop constructCPLopWithLeftTransposeRewrite() 
		throws HopsException, LopsException
	{
		Hop X = getInput().get(0).getInput().get(0); //guaranteed to exists
		Hop Y = getInput().get(1);
		
		//right vector transpose
		Lop tY = new Transform(Y.constructLops(), OperationTypes.Transpose, getDataType(), getValueType(), ExecType.CP);
		tY.getOutputParameters().setDimensions(Y.getDim2(), Y.getDim1(), getRowsInBlock(), getColsInBlock(), Y.getNnz());
		setLineNumbers(tY);
		
		//matrix mult
		Lop mult = new Binary(tY, X.constructLops(), Binary.OperationTypes.MATMULT, getDataType(), getValueType(), ExecType.CP);	
		mult.getOutputParameters().setDimensions(Y.getDim2(), X.getDim2(), getRowsInBlock(), getColsInBlock(), getNnz());
		setLineNumbers(mult);
		
		//result transpose (dimensions set outside)
		Lop out = new Transform(mult, OperationTypes.Transpose, getDataType(), getValueType(), ExecType.CP);
		
		return out;
	}
	
	/**
	 * 
	 * @return
	 * @throws HopsException
	 * @throws LopsException
	 */
	private Lop constructMapMultMRLopWithLeftTransposeRewrite() 
		throws HopsException, LopsException
	{
		Hop X = getInput().get(0).getInput().get(0); //guaranteed to exists
		Hop Y = getInput().get(1);
		
		//right vector transpose CP
		Lop tY = new Transform(Y.constructLops(), OperationTypes.Transpose, getDataType(), getValueType(), ExecType.CP);
		tY.getOutputParameters().setDimensions(Y.getDim2(), Y.getDim1(), getRowsInBlock(), getColsInBlock(), Y.getNnz());
		setLineNumbers(tY);
		
		//matrix mult
		
		// If number of columns is smaller than block size then explicit aggregation is not required.
		// i.e., entire matrix multiplication can be performed in the mappers.
		boolean needAgg = ( X.getDim1() <= 0 || X.getDim1() > X.getRowsInBlock() ); 
		boolean needPart = requiresPartitioning(MMultMethod.MAPMM_R, true); //R disregarding transpose rewrite
		
		//pre partitioning
		Lop dcinput = null;
		if( needPart ) {
			ExecType etPart = (OptimizerUtils.estimateSizeExactSparsity(Y.getDim2(), Y.getDim1(), OptimizerUtils.getSparsity(Y.getDim2(), Y.getDim1(), Y.getNnz())) 
					          < OptimizerUtils.getLocalMemBudget()) ? ExecType.CP : ExecType.MR; //operator selection
			dcinput = new DataPartition(tY, DataType.MATRIX, ValueType.DOUBLE, etPart, PDataPartitionFormat.COLUMN_BLOCK_WISE_N);
			dcinput.getOutputParameters().setDimensions(Y.getDim2(), Y.getDim1(), getRowsInBlock(), getColsInBlock(), Y.getNnz());
			setLineNumbers(dcinput);
		}
		else
			dcinput = tY;
		
		MapMult mapmult = new MapMult(dcinput, X.constructLops(), getDataType(), getValueType(), false, needPart, false);
		mapmult.getOutputParameters().setDimensions(Y.getDim2(), X.getDim2(), getRowsInBlock(), getColsInBlock(), getNnz());
		setLineNumbers(mapmult);
		
		//post aggregation 
		Lop mult = null;
		if( needAgg ) {
			Group grp = new Group(mapmult, Group.OperationTypes.Sort, getDataType(), getValueType());
			grp.getOutputParameters().setDimensions(Y.getDim2(), X.getDim2(), getRowsInBlock(), getColsInBlock(), getNnz());
			setLineNumbers(grp);
			
			Aggregate agg1 = new Aggregate(grp, HopsAgg2Lops.get(outerOp), getDataType(), getValueType(), ExecType.MR);
			agg1.getOutputParameters().setDimensions(Y.getDim2(), X.getDim2(), getRowsInBlock(), getColsInBlock(), getNnz());
			setLineNumbers(agg1);
			agg1.setupCorrectionLocation(CorrectionLocationType.NONE);  
			mult = agg1;
		}
		else
			mult = mapmult;
		
		//result transpose CP 
		Lop out = new Transform(mult, OperationTypes.Transpose, getDataType(), getValueType(), ExecType.CP);
		out.getOutputParameters().setDimensions(X.getDim2(), Y.getDim2(), getRowsInBlock(), getColsInBlock(), getNnz());
		
		return out;
	}

	/**
	 * 
	 * @return
	 * @throws HopsException
	 * @throws LopsException
	 */
	private Lop constructMMCJMRLopWithLeftTransposeRewrite() 
		throws HopsException, LopsException
	{
		Hop X = getInput().get(0).getInput().get(0); //guaranteed to exists
		Hop Y = getInput().get(1);
		
		//right vector transpose CP
		Lop tY = new Transform(Y.constructLops(), OperationTypes.Transpose, getDataType(), getValueType(), ExecType.CP);
		tY.getOutputParameters().setDimensions(Y.getDim2(), Y.getDim1(), getRowsInBlock(), getColsInBlock(), Y.getNnz());
		setLineNumbers(tY);
		
		//matrix multiply
		MMCJ mmcj = new MMCJ(tY, X.constructLops(), getDataType(), getValueType());
		mmcj.getOutputParameters().setDimensions(getDim1(), getDim2(), getRowsInBlock(), getColsInBlock(), getNnz());
		setLineNumbers(mmcj);
		
		Group grp = new Group(mmcj, Group.OperationTypes.Sort, getDataType(), getValueType());
		grp.getOutputParameters().setDimensions(getDim1(), getDim2(), getRowsInBlock(), getColsInBlock(), getNnz());
		setLineNumbers(grp);
		
		Aggregate agg1 = new Aggregate(grp, HopsAgg2Lops.get(outerOp), getDataType(), getValueType(), ExecType.MR);
		agg1.getOutputParameters().setDimensions(getDim1(), getDim2(), getRowsInBlock(), getColsInBlock(), getNnz());
		setLineNumbers(agg1);
		
		// aggregation uses kahanSum but the inputs do not have correction values
		agg1.setupCorrectionLocation(CorrectionLocationType.NONE);  

		
		//result transpose CP 
		Lop out = new Transform(agg1, OperationTypes.Transpose, getDataType(), getValueType(), ExecType.CP);
		out.getOutputParameters().setDimensions(X.getDim2(), Y.getDim2(), getRowsInBlock(), getColsInBlock(), getNnz());
		
		return out;
	}
	
	/**
	 * 
	 * @param method
	 * @return
	 */
	private boolean requiresAggregation(MMultMethod method) 
	{
		//worst-case assumption (for plan correctness)
		boolean ret = true;
		
		//right side cached (no agg if left has just one column block)
		if(  method == MMultMethod.MAPMM_R && getInput().get(0).getDim2() > 0 //known num columns
	         && getInput().get(0).getDim2() <= getInput().get(0).getColsInBlock() ) 
        {
            ret = false;
        }
        
		//left side cached (no agg if right has just one row block)
        if(  method == MMultMethod.MAPMM_L && getInput().get(1).getDim1() > 0 //known num rows
             && getInput().get(1).getDim1() <= getInput().get(1).getRowsInBlock() ) 
        {
       	    ret = false;
        }
        
        return ret;
	}
	
	/**
	 * 
	 * @param method
	 * @return
	 */
	private boolean requiresPartitioning(MMultMethod method, boolean rewrite) 
	{
		boolean ret = true; //worst-case
		Hop input1 = getInput().get(0);
		Hop input2 = getInput().get(1);
		
		//right side cached 
		if(  method == MMultMethod.MAPMM_R && input2.dimsKnown() ) { //known input size 
            ret = (input2.getDim1()*input2.getDim2() > DistributedCacheInput.PARTITION_SIZE);
            
            //conservative: do not apply partitioning if this forces t(X) into separate job
            //if( !rewrite && input1 instanceof ReorgOp && ((ReorgOp)input1).getOp()==ReOrgOp.TRANSPOSE )
            //	ret = false;
        }
        
		//left side cached (no agg if right has just one row block)
		if(  method == MMultMethod.MAPMM_L && input1.dimsKnown() ) { //known input size 
            ret = (input1.getDim1()*input1.getDim2() > DistributedCacheInput.PARTITION_SIZE);
            
            //conservative: do not apply partitioning if this forces t(X) into separate job
            //if( !rewrite && input2 instanceof ReorgOp && ((ReorgOp)input2).getOp()==ReOrgOp.TRANSPOSE )
            //	ret = false;
        }
        
		return ret;
	}
	
	
	/**
	 * Estimates the memory footprint of MapMult operation depending on which input is put into distributed cache.
	 * This function is called by <code>optFindMMultMethod()</code> to decide the execution strategy, as well as by 
	 * piggybacking to decide the number of Map-side instructions to put into a single GMR job. 
	 */
	public static double footprintInMapper (long m1_rows, long m1_cols, long m1_rpb, long m1_cpb, long m2_rows, long m2_cols, long m2_rpb, long m2_cpb, int cachedInputIndex, boolean pmm) 
	{
		// If the size of one input is small, choose a method that uses distributed cache
		// NOTE: be aware of output size because one input block might generate many output blocks
		double m1Size = OptimizerUtils.estimateSize(m1_rows, m1_cols);
		double m2Size = OptimizerUtils.estimateSize(m2_rows, m2_cols);
		double m1BlockSize = OptimizerUtils.estimateSize(Math.min(m1_rows, m1_rpb), Math.min(m1_cols, m1_cpb));
		double m2BlockSize = OptimizerUtils.estimateSize(Math.min(m2_rows, m2_rpb), Math.min(m2_cols, m2_cpb));
		double m3m1OutSize = OptimizerUtils.estimateSize(Math.min(m1_rows, m1_rpb), m2_cols); //output per m1 block if m2 in cache
		double m3m2OutSize = OptimizerUtils.estimateSize(m1_rows, Math.min(m2_cols, m2_cpb)); //output per m2 block if m1 in cache
	
		double footprint = 0;
		if( pmm )
		{
			//permutation matrix multiply 
			//(one input block -> at most two output blocks)
			footprint = m1Size + 3*m2BlockSize; //in+2*out
		}
		else
		{
			//generic matrix multiply
			if ( cachedInputIndex == 1 ) {
				// left input (m1) is in cache
				footprint = m1Size+m2BlockSize+m3m2OutSize;
			}
			else {
				// right input (m2) is in cache
				footprint = m1BlockSize+m2Size+m3m1OutSize;
			}	
		}
		
		return footprint;
	}
	
	/**
	 * Optimization that chooses between two methods to perform matrix multiplication on map-reduce.
	 * 
	 * More details on the cost-model used: refer ICDE 2011 paper. 
	 */
	private static MMultMethod optFindMMultMethodMR ( long m1_rows, long m1_cols, long m1_rpb, long m1_cpb, 
			                                        long m2_rows, long m2_cols, long m2_rpb, long m2_cpb, 
			                                        MMTSJType mmtsj, ChainType chainType, boolean leftPMInput ) 
	{	
		double memBudget = MAPMULT_MEM_MULTIPLIER * OptimizerUtils.getRemoteMemBudgetMap(true);		
		
		// Step 0: check for forced mmultmethod
		if( FORCED_MMULT_METHOD !=null )
			return FORCED_MMULT_METHOD;
		
		// Step 1: check TSMM
		// If transpose self pattern and result is single block:
		// use specialized TSMM method (always better than generic jobs)
		if(    ( mmtsj == MMTSJType.LEFT && m2_cols>=0 && m2_cols <= m2_cpb )
			|| ( mmtsj == MMTSJType.RIGHT && m1_rows>=0 && m1_rows <= m1_rpb ) )
		{
			return MMultMethod.TSMM;
		}

		// Step 2: check MapMultChain
		// If mapmultchain pattern and result is a single block:
		// use specialized mapmult method
		if( OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES )
		{
			//matmultchain if dim2(X)<=blocksize and all vectors fit in mappers
			//(X: m1_cols x m1_rows, v: m1_rows x m2_cols, w: m1_cols x m2_cols) 
			if( chainType!=ChainType.NONE && m1_rows>=0 && m1_rows<= m1_rpb
				&& m2_cols>=0 && m2_cols<=m2_cpb )
			{
				if( chainType==ChainType.XtXv && m1_rows>=0 && m2_cols>=0 
					&& OptimizerUtils.estimateSize(m1_rows, m2_cols ) < memBudget )
				{
					return MMultMethod.MAPMM_CHAIN;
				}
				else if( chainType==ChainType.XtwXv && m1_rows>=0 && m2_cols>=0 && m1_cols>=0
					&&   OptimizerUtils.estimateSize(m1_rows, m2_cols ) 
					   + OptimizerUtils.estimateSize(m1_cols, m2_cols) < memBudget )
				{
					return MMultMethod.MAPMM_CHAIN;
				}
			}
		}
		
		// Step 3: check for PMM (permutation matrix needs to fit into mapper memory)
		// (needs to be checked before mapmult for consistency with removeEmpty compilation 
		double footprintPM1 = footprintInMapper(m1_rows, 1, m1_rpb, m1_cpb, m2_rows, m2_cols, m2_rpb, m2_cpb, 1, true);
		double footprintPM2 = footprintInMapper(m2_rows, 1, m1_rpb, m1_cpb, m2_rows, m2_cols, m2_rpb, m2_cpb, 1, true);
		if( (footprintPM1 < memBudget && m1_rows>=0 || footprintPM2 < memBudget && m2_rows>=0 ) 
			&& leftPMInput ) 
		{
			return MMultMethod.PMM;
		}
		
		// Step 4: check MapMult
		// If the size of one input is small, choose a method that uses distributed cache
		// (with awareness of output size because one input block might generate many output blocks)		
		// memory footprint if left input is put into cache
		double footprint1 = footprintInMapper(m1_rows, m1_cols, m1_rpb, m1_cpb, m2_rows, m2_cols, m2_rpb, m2_cpb, 1, false);
		// memory footprint if right input is put into cache
		double footprint2 = footprintInMapper(m1_rows, m1_cols, m1_rpb, m1_cpb, m2_rows, m2_cols, m2_rpb, m2_cpb, 2, false);		
		double m1Size = OptimizerUtils.estimateSize(m1_rows, m1_cols);
		double m2Size = OptimizerUtils.estimateSize(m2_rows, m2_cols);
		if (   (footprint1 < memBudget && m1_rows>=0 && m1_cols>=0)
			|| (footprint2 < memBudget && m2_rows>=0 && m2_cols>=0) ) 
		{
			//apply map mult if one side fits in remote task memory 
			//(if so pick smaller input for distributed cache)
			if( m1Size < m2Size && m1_rows>=0 && m1_cols>=0) //FIXME
				return MMultMethod.MAPMM_L;
			else
				return MMultMethod.MAPMM_R;
		}
		
		// Step 5: check for unknowns
		// If the dimensions are unknown at compilation time, simply assume 
		// the worst-case scenario and produce the most robust plan -- which is CPMM
		if ( m1_rows == -1 || m1_cols == -1 || m2_rows == -1 || m2_cols == -1 )
			return MMultMethod.CPMM;

		// Step 6: Decide CPMM vs RMM based on io costs
		long m1_nrb = (long) Math.ceil((double)m1_rows/m1_rpb); // number of row blocks in m1
		long m1_ncb = (long) Math.ceil((double)m1_cols/m1_cpb); // number of column blocks in m1
		long m2_ncb = (long) Math.ceil((double)m2_cols/m2_cpb); // number of column blocks in m2

		// TODO: we must factor in the "sparsity"
		double m1_size = m1_rows * m1_cols;
		double m2_size = m2_rows * m2_cols;
		double result_size = m1_rows * m2_cols;

		int numReducersRMM = OptimizerUtils.getNumReducers(true);
		int numReducersCPMM = OptimizerUtils.getNumReducers(false);
		
		// Estimate the cost of RMM
		// RMM phase 1
		double rmm_shuffle = (m2_ncb*m1_size) + (m1_nrb*m2_size);
		double rmm_io = m1_size + m2_size + result_size;
		double rmm_nred = Math.min( m1_nrb * m2_ncb, //max used reducers 
				                    numReducersRMM); //available reducers
		// RMM total costs
		double rmm_costs = (rmm_shuffle + rmm_io) / rmm_nred;
		
		// Estimate the cost of CPMM
		// CPMM phase 1
		double cpmm_shuffle1 = m1_size + m2_size;
		double cpmm_nred1 = Math.min( m1_ncb, //max used reducers 
                                      numReducersCPMM); //available reducers		
		double cpmm_io1 = m1_size + m2_size + cpmm_nred1 * result_size;
		// CPMM phase 2
		double cpmm_shuffle2 = cpmm_nred1 * result_size;
		double cpmm_io2 = cpmm_nred1 * result_size + result_size;			
		double cpmm_nred2 = Math.min( m1_nrb * m2_ncb, //max used reducers 
                                      numReducersCPMM); //available reducers		
		// CPMM total costs
		double cpmm_costs =  (cpmm_shuffle1+cpmm_io1)/cpmm_nred1  //cpmm phase1
		                    +(cpmm_shuffle2+cpmm_io2)/cpmm_nred2; //cpmm phase2
		
		//final mmult method decision 
		if ( cpmm_costs < rmm_costs ) 
			return MMultMethod.CPMM;
		else 
			return MMultMethod.RMM;
	}

	private static MMultMethod optFindMMultMethodSpark( long m1_rows, long m1_cols, long m1_rpb, long m1_cpb, 
            long m2_rows, long m2_cols, long m2_rpb, long m2_cpb, MMTSJType mmtsj ) 
	{	
		//note: for spark we are taking half of the available budget since we do an in-memory partitioning
		double memBudget = MAPMULT_MEM_MULTIPLIER * SparkExecutionContext.getBroadcastMemoryBudget() / 2;		

		// Step 1: check TSMM
		// If transpose self pattern and result is single block:
		// use specialized TSMM method (always better than generic jobs)
		if(    ( mmtsj == MMTSJType.LEFT && m2_cols>=0 && m2_cols <= m2_cpb )
			|| ( mmtsj == MMTSJType.RIGHT && m1_rows>=0 && m1_rows <= m1_rpb ) )
		{
			return MMultMethod.TSMM;
		}
		
		// Step 2: check MapMult
		// If the size of one input is small, choose a method that uses broadcast variables
		// (currently we only apply this if a single output block)
		double footprint1 = footprintInMapper(m1_rows, m1_cols, m1_rpb, m1_cpb, m2_rows, m2_cols, m2_rpb, m2_cpb, 1, false);
		double footprint2 = footprintInMapper(m1_rows, m1_cols, m1_rpb, m1_cpb, m2_rows, m2_cols, m2_rpb, m2_cpb, 2, false);		
		double m1Size = OptimizerUtils.estimateSize(m1_rows, m1_cols);
		double m2Size = OptimizerUtils.estimateSize(m2_rows, m2_cols);
		
		if (   (footprint1 < memBudget && m1_rows>=0 && m1_cols>=0 && m1_rows<=m1_rpb)
			|| (footprint2 < memBudget && m2_rows>=0 && m2_cols>=0 && m2_cols<=m2_cpb) ) 
		{
			//apply map mult if one side fits in remote task memory 
			//(if so pick smaller input for distributed cache)
			if( m1Size < m2Size && m1_rows>=0 && m1_cols>=0) //FIXME
				return MMultMethod.MAPMM_L;
			else
				return MMultMethod.MAPMM_R;
		}
		
		// Step 3: fallback strategy MMCJ
		return MMultMethod.CPMM;
	}
	
	@Override
	public SQLLops constructSQLLOPs() throws HopsException {
	
		if(this.getSqlLops() == null)
		{
			if(this.getInput().size() != 2)
				throw new HopsException(this.printErrorLocation() + "In AggBinary Hop, The binary aggregation hop must have two inputs");
			
			//Check whether this is going to be an Insert or With
			GENERATES gen = determineGeneratesFlag();
			
			Hop hop1 = this.getInput().get(0);
			Hop hop2 = this.getInput().get(1);
			
			
			if(this.isMatrixMultiply())
			{
				SQLLops sqllop = getMatrixMultSQLLOP(gen);
				sqllop.set_properties(getProperties(hop1, hop2));
				this.setSqlLops(sqllop);
			}
			else
			{
				SQLLops sqllop = new SQLLops(this.getName(),
										gen,
										hop1.constructSQLLOPs(),
										hop2.constructSQLLOPs(),
										this.getValueType(), this.getDataType());
	
				String sql = getSQLSelectCode(hop1, hop2);
			
				sqllop.set_sql(sql);
				sqllop.set_properties(getProperties(hop1, hop2));
			
				this.setSqlLops(sqllop);
			}
			this.setVisited(VisitStatus.DONE);
		}
		return this.getSqlLops();
	}
	
	private SQLLopProperties getProperties(Hop hop1, Hop hop2)
	{
		SQLLopProperties prop = new SQLLopProperties();
		JOINTYPE join = JOINTYPE.FULLOUTERJOIN;
		AGGREGATIONTYPE agg = AGGREGATIONTYPE.NONE;
		
		if(innerOp == OpOp2.MULT || innerOp == OpOp2.AND || outerOp == AggOp.PROD)
			join = JOINTYPE.INNERJOIN;
		else if(innerOp == OpOp2.DIV)
			join = JOINTYPE.LEFTJOIN;
		
		//TODO: PROD
		if(outerOp == AggOp.SUM || outerOp == AggOp.TRACE)
			agg = AGGREGATIONTYPE.SUM;
		else if(outerOp == AggOp.MAX)
			agg = AGGREGATIONTYPE.MAX;
		else if(outerOp == AggOp.MIN)
			agg = AGGREGATIONTYPE.MIN;
		
		prop.setAggType(agg);
		prop.setJoinType(join);
		prop.setOpString(Hop.HopsAgg2String.get(outerOp) + "(" 
				+ hop1.getSqlLops().get_tableName() + " "
				+ Hop.HopsOpOp2String.get(innerOp) + " "
				+ hop1.getSqlLops().get_tableName() + ")");
		return prop;
	}
	
	@SuppressWarnings("unused")
	private SQLSelectStatement getSQLSelect(Hop hop1, Hop hop2) throws HopsException
	{
		if(!(hop1.getSqlLops().getDataType() == DataType.MATRIX && hop2.getSqlLops().getDataType() == DataType.MATRIX))
			throw new HopsException(this.printErrorLocation() + "In AggBinary Hop, Aggregates only work for two matrices");
		
		boolean isvalid = Hop.isSupported(this.innerOp);
		if(!isvalid)
			throw new HopsException(this.printErrorLocation() + "In AggBinary Hop, This operation is not supported for SQL Select");
		
		boolean isfunc = Hop.isFunction(this.innerOp);
		
		String inner_opr = SQLLops.OpOp2ToString(innerOp);

		SQLSelectStatement stmt = new SQLSelectStatement();

		JOINTYPE t = JOINTYPE.FULLOUTERJOIN;
		if(innerOp == OpOp2.MULT || innerOp == OpOp2.AND)
			t = JOINTYPE.INNERJOIN;
		else if(innerOp == OpOp2.DIV)
			t = JOINTYPE.LEFTJOIN;
		
		SQLJoin join = new SQLJoin();
		join.setJoinType(t);
		join.setTable1(new SQLTableReference(SQLLops.addQuotes(hop1.getSqlLops().get_tableName()), SQLLops.ALIAS_A));
		join.setTable2(new SQLTableReference(SQLLops.addQuotes(hop2.getSqlLops().get_tableName()), SQLLops.ALIAS_B));

		stmt.setTable(join);
		
		String inner = isfunc ? String.format(SQLLops.FUNCTIONOP_PART, inner_opr) :
				String.format(SQLLops.BINARYOP_PART, inner_opr);
		
		if(this.outerOp == AggOp.TRACE)
		{
			join.getConditions().add(new SQLCondition("alias_a.row = alias_a.col"));
			join.getConditions().add(new SQLCondition(BOOLOP.AND, "alias_a.row = alias_b.col"));
			join.getConditions().add(new SQLCondition(BOOLOP.AND, "alias_a.col = alias_b.row"));
			
			stmt.getColumns().add("coalesce(alias_a.row, alias_b.row) AS row");
			stmt.getColumns().add("coalesce(alias_a.col, alias_b.col) AS col");
			stmt.getColumns().add("SUM(" + inner + ")");
		}
		else if(this.outerOp == AggOp.SUM)
		{
			join.getConditions().add(new SQLCondition("alias_a.col = alias_b.row"));
			stmt.getColumns().add("alias_a.row AS row");
			stmt.getColumns().add("alias_b.col AS col");
			stmt.getColumns().add("SUM(" + inner + ")");
			
			stmt.getGroupBys().add("alias_a.row");
			stmt.getGroupBys().add("alias_b.col");
		}
		else
		{
			String outer = Hop.HopsAgg2String.get(this.outerOp);

			join.getConditions().add(new SQLCondition("alias_a.col = alias_b.row"));
			stmt.getColumns().add("alias_a.row AS row");
			stmt.getColumns().add("alias_b.col AS col");
			stmt.getColumns().add(outer);
			
			stmt.getGroupBys().add("alias_a.row");
			stmt.getGroupBys().add("alias_b.col");
		}

		return stmt;
	}
	
	private String getSQLSelectCode(Hop hop1, Hop hop2) throws HopsException
	{
		if(!(hop1.getSqlLops().getDataType() == DataType.MATRIX && hop2.getSqlLops().getDataType() == DataType.MATRIX))
			throw new HopsException(this.printErrorLocation() + "in AggBinary Hop, error in getSQLSelectCode() -- Aggregates only work for two matrices");
		
		//min, max, log, quantile, interquantile and iqm cannot be done that way
		boolean isvalid = Hop.isSupported(this.innerOp);

		//But min, max and log can still be done
		boolean isfunc = Hop.isFunction(this.innerOp);
		
		String inner_opr = SQLLops.OpOp2ToString(innerOp);
		
		if(isvalid)
		{
			//String for the inner operation
			String inner = isfunc ? String.format(SQLLops.FUNCTIONOP_PART, inner_opr)
					:
					String.format(SQLLops.BINARYOP_PART, inner_opr);
			
			String sql = null;
			String join = SQLLops.FULLOUTERJOIN;
			if(innerOp == OpOp2.MULT || innerOp == OpOp2.AND)
				join = SQLLops.JOIN;
			else if(innerOp == OpOp2.DIV)
				join = SQLLops.LEFTJOIN;
			
			if(this.outerOp == AggOp.PROD)
			{
				// This is only a temporary solution.
				// Idea is that ln(x1 * x2 * ... * xn) = ln(x1) + ln(x2) + ... + ln(xn)
				// So that x1 * x2 * ... * xn = exp( ln(x1) + ln(x2) + ... + ln(xn) )
				// Which is EXP(SUM(LN(v)))
				
				sql = String.format(SQLLops.BINARYPROD,
						inner, hop1.getSqlLops().get_tableName(), hop2.getSqlLops().get_tableName());
			}
			//Special case for trace because it needs a special SELECT
			else if(this.outerOp == AggOp.TRACE)
			{
				sql = String.format(SQLLops.AGGTRACEOP, inner, hop1.getSqlLops().get_tableName(), join, hop2.getSqlLops().get_tableName());
			}
			//Should be handled before
			else if(this.outerOp == AggOp.SUM)
			{
				//sql = String.format(SQLLops.AGGSUMOP, inner, hop1.getSqlLops().get_tableName(), join, hop2.getSqlLops().get_tableName());
				//sql = getMatrixMultSQLString(inner, hop1.getSqlLops().get_tableName(), hop2.getSqlLops().get_tableName());
			}
			//Here the outerOp is just appended, it can only be min or max
			else
			{
				String outer = Hop.HopsAgg2String.get(this.outerOp);
				sql = String.format(SQLLops.AGGBINOP, outer, inner, 
					hop1.getSqlLops().get_tableName(), join, hop2.getSqlLops().get_tableName());
			}
			return sql;
		}
		throw new HopsException(this.printErrorLocation() + "In AggBinary Hop, error in getSQLSelectCode() -- This operation is not supported");
	}
	
	private SQLLops getPart1SQLLop(String operation, String op1, String op2, long size, SQLLops input1, SQLLops input2)
	{		
		String where = SQLLops.ALIAS_A + ".col <= " + size
		+ " AND " + SQLLops.ALIAS_B + ".row <= " + size;
		String sql = String.format(SQLLops.SPLITAGGSUMOP, operation, op1, SQLLops.JOIN, op2, where);
		SQLLops lop = new SQLLops("part1_" + this.getHopID(), GENERATES.SQL, input1, input2, ValueType.DOUBLE, DataType.MATRIX);
		SQLLopProperties prop = new SQLLopProperties();
		prop.setAggType(AGGREGATIONTYPE.SUM);
		prop.setJoinType(JOINTYPE.INNERJOIN);
		prop.setOpString("Part 1 of matrix mult");
		lop.set_properties(prop);
		lop.set_sql(sql);
		return lop;
	}
	
	private SQLLops getPart2SQLLop(String operation, String op1, String op2, long size, SQLLops input1, SQLLops input2)
	{
		String where = SQLLops.ALIAS_A + ".col > " + size
		+ " AND " + SQLLops.ALIAS_B + ".row > " + size;
		String sql = String.format(SQLLops.SPLITAGGSUMOP, operation, op1, SQLLops.JOIN, op2, where);
		SQLLops lop = new SQLLops("part2_" + this.getHopID(), GENERATES.SQL, input1, input2, ValueType.DOUBLE, DataType.MATRIX);
		SQLLopProperties prop = new SQLLopProperties();
		prop.setAggType(AGGREGATIONTYPE.SUM);
		prop.setJoinType(JOINTYPE.INNERJOIN);
		prop.setOpString("Part 2 of matrix mult");
		lop.set_properties(prop);
		lop.set_sql(sql);
		return lop;
	}
	
	private SQLLops getMatrixMultSQLLOP(GENERATES flag) throws HopsException
	{
		Hop hop1 = this.getInput().get(0);
		Hop hop2 = this.getInput().get(1);
		
		boolean m_large = hop1.getDim1() > SQLLops.HMATRIXSPLIT;
		boolean k_large = hop1.getDim2() > SQLLops.VMATRIXSPLIT;
		boolean n_large = hop2.getDim2() > SQLLops.HMATRIXSPLIT;
		
		String name = this.getName() + "_" + this.getHopID();
		
		String inner_opr = SQLLops.OpOp2ToString(innerOp);
		String operation = String.format(SQLLops.BINARYOP_PART, inner_opr);
		
		SQLLops input1 = hop1.constructSQLLOPs();
		SQLLops input2 = hop2.constructSQLLOPs();
		
		String i1 = input1.get_tableName();
		String i2 = input2.get_tableName();
		
		if(!SPLITLARGEMATRIXMULT || (!m_large && !k_large && !n_large))
		{
			String sql = String.format(SQLLops.AGGSUMOP, operation, i1, SQLLops.JOIN, i2);
			SQLLops lop = new SQLLops(name, flag, hop1.constructSQLLOPs(), hop2.constructSQLLOPs(), ValueType.DOUBLE, DataType.MATRIX);
			lop.set_sql(sql);
			return lop;
		}
		else if(m_large)
		{
			StringBuilder sb = new StringBuilder();
			//Split first matrix horizontally

			long total = 0;
			for(long s = SQLLops.HMATRIXSPLIT; s <= hop1.getDim1(); s += SQLLops.HMATRIXSPLIT)
			{
				String where = SQLLops.ALIAS_A + ".row BETWEEN " + (s - SQLLops.HMATRIXSPLIT + 1) + " AND " + s;
				sb.append(String.format(SQLLops.SPLITAGGSUMOP, operation, i1, SQLLops.JOIN, i2, where));
				
				total = s;
				if(total < hop1.getDim1())
					sb.append(" \r\nUNION ALL \r\n");
			}
			if(total < hop1.getDim1())
			{
				String where = SQLLops.ALIAS_A + ".row BETWEEN " + (total + 1) + " AND " + hop1.getDim1();
				sb.append(String.format(SQLLops.SPLITAGGSUMOP, operation, i1, SQLLops.JOIN, i2, where));
			}
			String sql = sb.toString();
			SQLLops lop = new SQLLops(name, flag, hop1.getSqlLops(), hop2.getSqlLops(), ValueType.DOUBLE, DataType.MATRIX);
			lop.set_sql(sql);
			return lop;
		}
		else if(k_large)
		{
			//The parts are both DML and have the same input, so it cannot be SQL even though in the HOPs DAg it might have only
			// one output
			/*if(input1.get_flag() == GENERATES.SQL)
				input1.set_flag(GENERATES.DML);
			if(input2.get_flag() == GENERATES.SQL)
				input2.set_flag(GENERATES.DML);
			*/
			SQLLops h1 = getPart1SQLLop(operation, i1, i2, hop1.getDim2() / 2, input1, input2);
			SQLLops h2 = getPart2SQLLop(operation, i1, i2, hop1.getDim2() / 2, input1, input2);
			
			String p1 = SQLLops.addQuotes(h1.get_tableName());
			String p2 = SQLLops.addQuotes(h2.get_tableName());
			
			String sql = "SELECT coalesce(" + p1 + ".row, " + p2 + ".row) AS row, coalesce(" + p1 + ".col, " + p2 + ".col) AS col, "
				+ "coalesce(" + p1 + ".value,0) + coalesce(" + p2 + ".value,0) AS value FROM " + p1 + " FULL OUTER JOIN " + p2
				+ " ON " + p1 + ".row = " + p2 + ".row AND " + p1 + ".col = " + p2 + ".col";

			SQLLops lop = new SQLLops(name, flag, h1, h2, ValueType.DOUBLE, DataType.MATRIX);
			lop.set_sql(sql);
			return lop;
		}
		else
		{
			String sql = String.format(SQLLops.AGGSUMOP, operation, i1, SQLLops.JOIN, i2);
			SQLLops lop = new SQLLops(name, flag, hop1.getSqlLops(), hop2.getSqlLops(), ValueType.DOUBLE, DataType.MATRIX);
			lop.set_sql(sql);
			return lop;
		}
	}
	
	@SuppressWarnings("unused")
	private String getMatrixMultSQLString(String operation, String op1, String op2)
	{
		Hop hop1 = this.getInput().get(0);
		Hop hop2 = this.getInput().get(1);
		
		boolean m_large = hop1.getDim1() > SQLLops.HMATRIXSPLIT;
		boolean k_large = hop1.getDim2() > SQLLops.VMATRIXSPLIT;
		boolean n_large = hop2.getDim2() > SQLLops.HMATRIXSPLIT;
		
		if(!SPLITLARGEMATRIXMULT || (!m_large && !k_large && !n_large))
			return String.format(SQLLops.AGGSUMOP, operation, op1, SQLLops.JOIN, op2);
		else
		{
			StringBuilder sb = new StringBuilder();
			//Split first matrix horizontally
			if(m_large)
			{
				long total = 0;
				for(long s = SQLLops.HMATRIXSPLIT; s <= hop1.getDim1(); s += SQLLops.HMATRIXSPLIT)
				{
					String where = SQLLops.ALIAS_A + ".row BETWEEN " + (s - SQLLops.HMATRIXSPLIT + 1) + " AND " + s;
					sb.append(String.format(SQLLops.SPLITAGGSUMOP, operation, op1, SQLLops.JOIN, op2, where));
					
					total = s;
					if(total < hop1.getDim1())
						sb.append(" \r\nUNION ALL \r\n");
				}
				if(total < hop1.getDim1())
				{
					String where = SQLLops.ALIAS_A + ".row BETWEEN " + (total + 1) + " AND " + hop1.getDim1();
					sb.append(String.format(SQLLops.SPLITAGGSUMOP, operation, op1, SQLLops.JOIN, op2, where));
				}
				return sb.toString();
			}
			//Split first matrix vertically and second matrix horizontally
			else if(k_large)
			{
				long middle = hop1.getDim2() / 2;
				
				String where1 = SQLLops.ALIAS_A + ".col <= " + middle
				+ " AND " + SQLLops.ALIAS_B + ".row <= " + middle;
				
				String where2 = SQLLops.ALIAS_A + ".col > " + middle
				+ " AND " + SQLLops.ALIAS_B + ".row > " + middle;
				
				sb.append("\r\nWITH part1 AS ( " + String.format(SQLLops.SPLITAGGSUMOP, operation, op1, SQLLops.JOIN, op2, where1) + "),\r\n");						
				sb.append("part2 AS ( " + String.format(SQLLops.SPLITAGGSUMOP, operation, op1, SQLLops.JOIN, op2, where2) + ")\r\n");
				sb.append("SELECT coalesce(part1.row, part2.row) AS row, coalesce(part1.col, part2.col) AS col, "
						+ "coalesce(part1.value,0) + coalesce(part2.value,0) AS value FROM part1 FULL OUTER JOIN part2 "
						+ "ON part1.row = part2.row AND part1.col = part2.col");
				
				return sb.toString();
				//TODO split
				//return String.format(SQLLops.AGGSUMOP, operation, op1, join, op2);
			}
			//Split second matrix vertically
			else
			{
				//TODO split
				return String.format(SQLLops.AGGSUMOP, operation, op1, SQLLops.JOIN, op2);
			}
		}
	}

	@Override
	public void refreshSizeInformation()
	{
		Hop input1 = getInput().get(0);
		Hop input2 = getInput().get(1);
		
		if( isMatrixMultiply() )
		{
			setDim1(input1.getDim1());
			setDim2(input2.getDim2());
		}
	}
	
	@Override
	public Object clone() throws CloneNotSupportedException 
	{
		AggBinaryOp ret = new AggBinaryOp();	
		
		//copy generic attributes
		ret.clone(this, false);
		
		//copy specific attributes
		ret.innerOp = innerOp;
		ret.outerOp = outerOp;		
		ret._hasLeftPMInput = _hasLeftPMInput;
		
		return ret;
	}
	
	@Override
	public boolean compare( Hop that )
	{
		if( that._kind!=Kind.AggBinaryOp )
			return false;
		
		AggBinaryOp that2 = (AggBinaryOp)that;
		return (   innerOp == that2.innerOp
				&& outerOp == that2.outerOp
				&& getInput().get(0) == that2.getInput().get(0)
				&& getInput().get(1) == that2.getInput().get(1)
				&& _hasLeftPMInput == that2._hasLeftPMInput);
	}
}
