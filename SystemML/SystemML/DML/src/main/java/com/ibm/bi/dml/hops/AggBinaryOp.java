/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops;

import com.ibm.bi.dml.lops.Aggregate;
import com.ibm.bi.dml.lops.BinaryCP;
import com.ibm.bi.dml.lops.Group;
import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.lops.LopsException;
import com.ibm.bi.dml.lops.MMCJ;
import com.ibm.bi.dml.lops.MMRJ;
import com.ibm.bi.dml.lops.MMTSJ;
import com.ibm.bi.dml.lops.MapMult;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.MMTSJ.MMTSJType;
import com.ibm.bi.dml.lops.PartialAggregate.CorrectionLocationType;
import com.ibm.bi.dml.lops.Transform;
import com.ibm.bi.dml.lops.Transform.OperationTypes;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.sql.sqllops.SQLCondition;
import com.ibm.bi.dml.sql.sqllops.SQLJoin;
import com.ibm.bi.dml.sql.sqllops.SQLLopProperties;
import com.ibm.bi.dml.sql.sqllops.SQLLops;
import com.ibm.bi.dml.sql.sqllops.SQLSelectStatement;
import com.ibm.bi.dml.sql.sqllops.SQLTableReference;
import com.ibm.bi.dml.sql.sqllops.SQLCondition.BOOLOP;
import com.ibm.bi.dml.sql.sqllops.SQLLopProperties.AGGREGATIONTYPE;
import com.ibm.bi.dml.sql.sqllops.SQLLopProperties.JOINTYPE;
import com.ibm.bi.dml.sql.sqllops.SQLLops.GENERATES;


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
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	public static final double MVMULT_MEM_MULTIPLIER = 1.0;
	
	private OpOp2 innerOp;
	private AggOp outerOp;

	private enum MMultMethod { 
		CPMM,     //cross-product matrix multiplication
		RMM,      //replication matrix multiplication
		MAPMULT_L,  //map-side matrix-matrix multiplication using distributed cache, for left input
		MAPMULT_R,  //map-side matrix-matrix multiplication using distributed cache, for right input
		TSMM,     //transpose-self matrix multiplication
		CP        //in-memory matrix multiplication
	};
	
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
	
	public boolean isMatrixMultiply () {
		return ( this.innerOp == OpOp2.MULT && this.outerOp == AggOp.SUM );			
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
		if (get_lops() == null) {
			if ( isMatrixMultiply() ) 
			{
				ExecType et = optFindExecType();
				MMTSJType mmtsj = checkTransposeSelf();
				
				if ( et == ExecType.CP ) 
				{
					Lop matmultCP = null;
					if( mmtsj == MMTSJType.NONE ) //CP MM
					{
						if( isLeftTransposeRewriteApplicable(true) )
							matmultCP = constructCPLopWithLeftTransposeRewrite();
						else
							matmultCP = new BinaryCP(getInput().get(0).constructLops(),getInput().get(1).constructLops(), 
												 BinaryCP.OperationTypes.MATMULT, get_dataType(), get_valueType());
					}
					else //CP TSMM
					{
						matmultCP = new MMTSJ(getInput().get((mmtsj==MMTSJType.LEFT)?1:0).constructLops(),
								              get_dataType(), get_valueType(),et, mmtsj);
					}
					
					matmultCP.getOutputParameters().setDimensions(get_dim1(), get_dim2(), get_rows_in_block(), get_cols_in_block(), getNnz());
					matmultCP.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
					set_lops(matmultCP);
				}
				else if ( et == ExecType.MR ) {
				
					MMultMethod method = optFindMMultMethod ( 
								getInput().get(0).get_dim1(), getInput().get(0).get_dim2(), 
								getInput().get(0).get_rows_in_block(), getInput().get(0).get_cols_in_block(),    
								getInput().get(1).get_dim1(), getInput().get(1).get_dim2(), 
								getInput().get(1).get_rows_in_block(), getInput().get(1).get_cols_in_block(),
								mmtsj);
					//System.out.println("Method = " + method);
					
					if ( method == MMultMethod.MAPMULT_L || method == MMultMethod.MAPMULT_R ) 
					{
						if( method == MMultMethod.MAPMULT_R && isLeftTransposeRewriteApplicable(false) )
						{
							set_lops( constructMRLopWithLeftTransposeRewrite() );
						}
						else //GENERAL CASE
						{
							//TODO revisit once it is taken into account in 'optFindMMultMethod'
							//if ( partitionVectorInDistCache(getInput().get(1)._dim1, getInput().get(1)._dim2) ) {
							//	smallMatrix = new DataPartition(getInput().get(1).constructLops(), get_dataType(), get_valueType());
							//}
							
							// If number of columns is smaller than block size then explicit aggregation is not required.
							// i.e., entire matrix multiplication can be performed in the mappers.
							boolean needAgg = requiresAggregation(method); 
							boolean outputEmptyBlocks = !hasOnlyCPConsumers(); 
							
							MapMult mvmult = new MapMult(getInput().get(0).constructLops(), getInput().get(1).constructLops(), 
									                     get_dataType(), get_valueType(), (method==MMultMethod.MAPMULT_R), outputEmptyBlocks);
							mvmult.getOutputParameters().setDimensions(get_dim1(), get_dim2(), get_rows_in_block(), get_cols_in_block(), getNnz());
							
							if (needAgg) {
								Group grp = new Group(mvmult, Group.OperationTypes.Sort, get_dataType(), get_valueType());
								Aggregate agg1 = new Aggregate(grp, HopsAgg2Lops.get(outerOp), get_dataType(), get_valueType(), ExecType.MR);
								
								grp.getOutputParameters().setDimensions(get_dim1(), get_dim2(), get_rows_in_block(), get_cols_in_block(), getNnz());
								agg1.getOutputParameters().setDimensions(get_dim1(), get_dim2(), get_rows_in_block(), get_cols_in_block(), getNnz());
								
								agg1.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
								
								// aggregation uses kahanSum but the inputs do not have correction values
								agg1.setupCorrectionLocation(CorrectionLocationType.NONE);  
								
								set_lops(agg1);
							}
							else {
								set_lops(mvmult);
							}
						}
					}
					else if ( method == MMultMethod.CPMM ) {
						MMCJ mmcj = new MMCJ(
								getInput().get(0).constructLops(), getInput().get(1)
										.constructLops(), get_dataType(), get_valueType());
						mmcj.getOutputParameters().setDimensions(get_dim1(), get_dim2(), 
								get_rows_in_block(), get_cols_in_block(), getNnz());
						
						mmcj.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
						
						Group grp = new Group(
								mmcj, Group.OperationTypes.Sort, get_dataType(), get_valueType());
						grp.getOutputParameters().setDimensions(get_dim1(), get_dim2(), 
								get_rows_in_block(), get_cols_in_block(), getNnz());
						
						grp.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
						
						Aggregate agg1 = new Aggregate(
								grp, HopsAgg2Lops.get(outerOp), get_dataType(), get_valueType(), ExecType.MR);
						agg1.getOutputParameters().setDimensions(get_dim1(), get_dim2(), 
								get_rows_in_block(), get_cols_in_block(), getNnz());
						
						agg1.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
						
						// aggregation uses kahanSum but the inputs do not have correction values
						agg1.setupCorrectionLocation(CorrectionLocationType.NONE);  
						
						set_lops(agg1);
					}
					else if (method == MMultMethod.RMM ) {
						MMRJ rmm = new MMRJ(
								getInput().get(0).constructLops(), getInput().get(1)
								.constructLops(), get_dataType(), get_valueType());
						
						rmm.getOutputParameters().setDimensions(get_dim1(), get_dim2(),
								get_rows_in_block(), get_cols_in_block(), getNnz());
						
						rmm.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
						
						set_lops(rmm);
					}
					else if( method == MMultMethod.TSMM )
					{
						MMTSJ tsmm = new MMTSJ(getInput().get((mmtsj==MMTSJType.LEFT)?1:0).constructLops(),
								              get_dataType(), get_valueType(),et, mmtsj);
						tsmm.getOutputParameters().setDimensions(get_dim1(), get_dim2(),
								get_rows_in_block(), get_cols_in_block(), getNnz());
						tsmm.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
						
						Aggregate agg1 = new Aggregate(
								tsmm, HopsAgg2Lops.get(outerOp), get_dataType(), get_valueType(), ExecType.MR);
						agg1.getOutputParameters().setDimensions(get_dim1(), get_dim2(), 
								get_rows_in_block(), get_cols_in_block(), getNnz());
						agg1.setupCorrectionLocation(CorrectionLocationType.NONE); // aggregation uses kahanSum but the inputs do not have correction values
						agg1.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
						
						set_lops(agg1);
					}
					else {
						throw new HopsException(this.printErrorLocation() + "Invalid Matrix Mult Method (" + method + ") while constructing lops.");
					}
				}
			} 
			else  {
				throw new HopsException(this.printErrorLocation() + "Invalid operation in AggBinary Hop, aggBin(" + innerOp + "," + outerOp + ") while constructing lops.");
			}
		}
		
		return get_lops();
	}

	@Override
	public String getOpString() {
		String s = new String("");
		s += "ab(" + HopsAgg2String.get(outerOp) + HopsOpOp2String.get(innerOp)+")";
		return s;
	}

	public void printMe() throws HopsException {
		if (LOG.isDebugEnabled()){
			if (get_visited() != VISIT_STATUS.DONE) {
				super.printMe();
				LOG.debug("  InnerOperation: " + innerOp);
				LOG.debug("  OuterOperation: " + outerOp);
				for (Hop h : getInput()) {
					h.printMe();
				}
				;
			}
			set_visited(VISIT_STATUS.DONE);
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
					double sp1 = (input1.getNnz()>0) ? OptimizerUtils.getSparsity(input1.get_dim1(), input1.get_dim2(), input1.getNnz()) : 1.0;
					double sp2 = (input2.getNnz()>0) ? OptimizerUtils.getSparsity(input2.get_dim1(), input2.get_dim2(), input2.getNnz()) : 1.0;
					sparsity = OptimizerUtils.getMatMultSparsity(sp1, sp2, input1.get_dim1(), input1.get_dim2(), input2.get_dim2(), true);	
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
			ret[0] = mc[0].get_rows();
			ret[1] = mc[1].get_cols();
			double sp1 = (mc[0].getNonZeros()>0) ? OptimizerUtils.getSparsity(mc[0].get_rows(), mc[0].get_cols(), mc[0].getNonZeros()) : 1.0; 
			double sp2 = (mc[1].getNonZeros()>0) ? OptimizerUtils.getSparsity(mc[1].get_rows(), mc[1].get_cols(), mc[1].getNonZeros()) : 1.0; 			
			ret[2] = (long) ( ret[0] * ret[1] * OptimizerUtils.getMatMultSparsity(sp1, sp2, ret[0], mc[0].get_cols(), ret[1], true));
		}
		
		return ret;
	}
	
	
	private boolean isOuterProduct() {
		if ( getInput().get(0).isVector() && getInput().get(1).isVector() ) {
			if ( getInput().get(0).get_dim1() == 1 && getInput().get(0).get_dim1() > 1
					&& getInput().get(1).get_dim1() > 1 && getInput().get(1).get_dim2() == 1 )
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
			
			//mark for recompile (forever)
			if( OptimizerUtils.ALLOW_DYN_RECOMPILATION && !dimsKnown(true) && _etype==ExecType.MR )
				setRequiresRecompile();			
		}
		return _etype;
	}
	
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
	 * Function that determines whether or not to partition the vector that 
	 * is in distributed cache. Returns <code>true</code> if the estimated 
	 * size of the vector is greater than 80% of remote mapper's memory, and 
	 * returns <code>false</code> otherwise. 
	 */
	private static boolean partitionVectorInDistCache(long rows, long cols) {
		//return true;
		double vec_size = OptimizerUtils.estimateSize(rows, cols, 1.0);
		return ( vec_size > MVMULT_MEM_MULTIPLIER * OptimizerUtils.getRemoteMemBudget(true) );
	}
	
	/**
	 * Determines if the rewrite t(X)%*%Y -> t(t(Y)%*%X) is applicable
	 * and cost effective. Whenever X is a wide matrix and Y is a vector
	 * this has huge impact, because the transpose of X would dominate
	 * the entire operation costs.
	 * 
	 * @return
	 */
	private boolean isLeftTransposeRewriteApplicable(boolean CP)
	{
		boolean ret = false;
		Hop h1 = getInput().get(0);
		Hop h2 = getInput().get(1);
		
		if( CP ) //in-memory ba (implies input/output transpose can also be CP)
		{
			if( h1 instanceof ReorgOp && ((ReorgOp)h1).getOp()==ReOrgOp.TRANSPOSE )
			{
				long m = h1.get_dim1();
				long cd = h1.get_dim2();
				long n = h2.get_dim2();
				
				//check for known dimensions and cost for t(M) vs t(v) + t(tvM)
				if( m>0 && cd>0 && n>0 && (m*cd > (cd*n + m*n)) ) 
					ret = true;
			}
		}
		else //MR
		{
			if( h1 instanceof ReorgOp && ((ReorgOp)h1).getOp()==ReOrgOp.TRANSPOSE )
			{
				long m = h1.get_dim1();
				long cd = h1.get_dim2();
				long n = h2.get_dim2();
				
				//check for known dimensions and cost for t(M) vs t(v) + t(tvM)
				//(compared to CP, we explicitly check that new transposes fit in memory)
				//note: output size constraint for mapmult already checked by optfindmmultmethod
				if( m>0 && cd>0 && n>0 && (m*cd > (cd*n + m*n)) &&
					2 * OptimizerUtils.estimateSizeExactSparsity(cd, n, 1.0) <  OptimizerUtils.getLocalMemBudget() &&
					2 * OptimizerUtils.estimateSizeExactSparsity(m, n, 1.0) <  OptimizerUtils.getLocalMemBudget() &&
					OptimizerUtils.estimateSizeExactSparsity(cd, n, 1.0) < OptimizerUtils.getRemoteMemBudget(true) ) 
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
		Lop tY = new Transform(Y.constructLops(), OperationTypes.Transpose, get_dataType(), get_valueType(), ExecType.CP);
		tY.getOutputParameters().setDimensions(Y.get_dim2(), Y.get_dim1(), get_rows_in_block(), get_cols_in_block(), Y.getNnz());
		tY.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
		
		//matrix mult
		Lop mult = new BinaryCP(tY, X.constructLops(), BinaryCP.OperationTypes.MATMULT, get_dataType(), get_valueType());	
		mult.getOutputParameters().setDimensions(Y.get_dim2(), X.get_dim2(), get_rows_in_block(), get_cols_in_block(), getNnz());
		
		//result transpose (dimensions set outside)
		Lop out = new Transform(mult, OperationTypes.Transpose, get_dataType(), get_valueType(), ExecType.CP);
	
		return out;
	}
	
	/**
	 * 
	 * @return
	 * @throws HopsException
	 * @throws LopsException
	 */
	private Lop constructMRLopWithLeftTransposeRewrite() 
		throws HopsException, LopsException
	{
		Hop X = getInput().get(0).getInput().get(0); //guaranteed to exists
		Hop Y = getInput().get(1);
		
		//right vector transpose CP
		Lop tY = new Transform(Y.constructLops(), OperationTypes.Transpose, get_dataType(), get_valueType(), ExecType.CP);
		tY.getOutputParameters().setDimensions(Y.get_dim2(), Y.get_dim1(), get_rows_in_block(), get_cols_in_block(), Y.getNnz());
		tY.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
		
		//matrix mult
		Lop mult = null;
		// If number of columns is smaller than block size then explicit aggregation is not required.
		// i.e., entire matrix multiplication can be performed in the mappers.
		boolean needAgg = ( X.get_dim1() <= 0 || X.get_dim1() > X.get_rows_in_block() ); 
		
		MapMult mvmult = new MapMult(tY, X.constructLops(), get_dataType(), get_valueType(), false, false);
		mvmult.getOutputParameters().setDimensions(Y.get_dim2(), X.get_dim2(), get_rows_in_block(), get_cols_in_block(), getNnz());
		
		if (needAgg) {
			Group grp = new Group(mvmult, Group.OperationTypes.Sort, get_dataType(), get_valueType());
			grp.getOutputParameters().setDimensions(Y.get_dim2(), X.get_dim2(), get_rows_in_block(), get_cols_in_block(), getNnz());
			
			Aggregate agg1 = new Aggregate(grp, HopsAgg2Lops.get(outerOp), get_dataType(), get_valueType(), ExecType.MR);
			agg1.getOutputParameters().setDimensions(Y.get_dim2(), X.get_dim2(), get_rows_in_block(), get_cols_in_block(), getNnz());
			agg1.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
			agg1.setupCorrectionLocation(CorrectionLocationType.NONE);  
			mult= agg1;
		}
		else
			mult = mvmult;
		
		//result transpose CP 
		Lop out = new Transform(mult, OperationTypes.Transpose, get_dataType(), get_valueType(), ExecType.CP);
		out.getOutputParameters().setDimensions(X.get_dim2(), Y.get_dim2(), get_rows_in_block(), get_cols_in_block(), getNnz());
		
		return out;
	}
	
	/**
	 * 
	 * @return
	 * @throws HopsException 
	 */
	private boolean hasOnlyCPConsumers() 
		throws HopsException
	{
		boolean ret = true;
		for( Hop p : getParent() ) {
			p.optFindExecType(); //ensure exec type evaluated
			ret &= ( p.getExecType()==ExecType.CP 
					|| (p instanceof AggBinaryOp && ((AggBinaryOp)p).hasOnlyCPConsumers()))
					&& !(p instanceof FunctionOp || p instanceof DataOp ); //no function call or transient write
		}
			
		return ret;	
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
		if(  method == MMultMethod.MAPMULT_R && getInput().get(0).get_dim2() > 0 //known num columns
	         && getInput().get(0).get_dim2() <= getInput().get(0).get_cols_in_block() ) 
        {
            ret = false;
        }
        
		//left side cached (no agg if right has just one row block)
        if(  method == MMultMethod.MAPMULT_L && getInput().get(1).get_dim1() > 0 //known num rows
             && getInput().get(1).get_dim1() <= getInput().get(1).get_rows_in_block() ) 
        {
       	    ret = false;
        }
        
        return ret;
	}
	
	/**
	 * Estimates the memory footprint of MapMult operation depending on which input is put into distributed cache.
	 * This function is called by <code>optFindMMultMethod()</code> to decide the execution strategy, as well as by 
	 * piggybacking to decide the number of Map-side instructions to put into a single GMR job. 
	 */
	public static double footprintInMapper (long m1_rows, long m1_cols, long m1_rpb, long m1_cpb, long m2_rows, long m2_cols, long m2_rpb, long m2_cpb, int cachedInputIndex) {
		// If the size of one input is small, choose a method that uses distributed cache
		// NOTE: be aware of output size because one input block might generate many output blocks
		double m1Size = OptimizerUtils.estimateSize(m1_rows, m1_cols, 1.0);
		double m1BlockSize = OptimizerUtils.estimateSize(Math.min(m1_rows, m1_rpb), Math.min(m1_cols, m1_cpb), 1.0);
		double m2Size = OptimizerUtils.estimateSize(m2_rows, m2_cols, 1.0);
		double m2BlockSize = OptimizerUtils.estimateSize(Math.min(m2_rows, m2_rpb), Math.min(m2_cols, m2_cpb), 1.0);
		double m3m1OutSize = OptimizerUtils.estimateSize(Math.min(m1_rows, m1_rpb), m2_cols, 1.0); //output per m1 block if m2 in cache
		double m3m2OutSize = OptimizerUtils.estimateSize(m1_rows, Math.min(m2_cols, m2_cpb), 1.0); //output per m2 block if m1 in cache
	
		double footprint = 0;
		if ( cachedInputIndex == 1 ) {
			// left input (m1) is in cache
			footprint = m1Size+m2BlockSize+m3m2OutSize;
		}
		else {
			// right input (m2) is in cache
			footprint = m1BlockSize+m2Size+m3m1OutSize;
		}
		return footprint;
	}
	
	/**
	 * Optimization that chooses between two methods to perform matrix multiplication on map-reduce -- CPMM or RMM.
	 * 
	 * More details on the cost-model used: refer ICDE 2011 paper. 
	 */
	private static MMultMethod optFindMMultMethod ( long m1_rows, long m1_cols, long m1_rpb, long m1_cpb, long m2_rows, long m2_cols, long m2_rpb, long m2_cpb, MMTSJType mmtsj ) {
		
		// If transpose self pattern and result is single block:
		// use specialized TSMM method (always better than generic jobs)
		if(    ( mmtsj == MMTSJType.LEFT && m2_cols <= m2_cpb )
			|| ( mmtsj == MMTSJType.RIGHT && m1_rows <= m1_rpb ) )
		{
			return MMultMethod.TSMM;
		}

		// If the size of one input is small, choose a method that uses distributed cache
		// NOTE: be aware of output size because one input block might generate many output blocks
		
		// memory footprint if left input is put into cache
		double footprint1 = footprintInMapper(m1_rows, m1_cols, m1_rpb, m1_cpb, m2_rows, m2_cols, m2_rpb, m2_cpb, 1);
		// memory footprint if right input is put into cache
		double footprint2 = footprintInMapper(m1_rows, m1_cols, m1_rpb, m1_cpb, m2_rows, m2_cols, m2_rpb, m2_cpb, 2);
		
		double m1Size = OptimizerUtils.estimateSize(m1_rows, m1_cols, 1.0);
		double m2Size = OptimizerUtils.estimateSize(m2_rows, m2_cols, 1.0);
		double memBudget = MVMULT_MEM_MULTIPLIER * OptimizerUtils.getRemoteMemBudget(true);
		
		if ( footprint1 < memBudget || footprint2 < memBudget ) 
		{
			//apply map mult if one side fits in remote task memory 
			//(if so pick smaller input for distributed cache)
			if( m1Size < m2Size )
				return MMultMethod.MAPMULT_L;
			else
				return MMultMethod.MAPMULT_R;
		}
		
		// If the dimensions are unknown at compilation time, 
		// simply assume the worst-case scenario and produce the 
		// most robust plan -- which is CPMM
		if ( m1_rows == -1 || m1_cols == -1 || m2_rows == -1 || m2_cols == -1 )
			return MMultMethod.CPMM;

		int m1_nrb = (int) Math.ceil((double)m1_rows/m1_rpb); // number of row blocks in m1
		int m1_ncb = (int) Math.ceil((double)m1_cols/m1_cpb); // number of column blocks in m1
		int m2_ncb = (int) Math.ceil((double)m2_cols/m2_cpb); // number of column blocks in m2

		// TODO: we must factor in the "sparsity"
		double m1_size = m1_rows * m1_cols;
		double m2_size = m2_rows * m2_cols;
		double result_size = m1_rows * m2_cols;

		int numReducers = OptimizerUtils.getNumReducers(false);
		
		/* Estimate the cost of RMM */
		// RMM phase 1
		double rmm_shuffle = (m2_ncb*m1_size) + (m1_nrb*m2_size);
		double rmm_io = m1_size + m2_size + result_size;
		double rmm_nred = Math.min( m1_nrb * m2_ncb, //max used reducers 
				                    numReducers); //available reducers
		// RMM total costs
		double rmm_costs = (rmm_shuffle + rmm_io) / rmm_nred;
		
		/* Estimate the cost of CPMM */
		// CPMM phase 1
		double cpmm_shuffle1 = m1_size + m2_size;
		double cpmm_nred1 = Math.min( m1_ncb, //max used reducers 
                                      numReducers); //available reducers		
		double cpmm_io1 = m1_size + m2_size + cpmm_nred1 * result_size;
		// CPMM phase 2
		double cpmm_shuffle2 = cpmm_nred1 * result_size;
		double cpmm_io2 = cpmm_nred1 * result_size + result_size;			
		double cpmm_nred2 = Math.min( m1_nrb * m2_ncb, //max used reducers 
                                      numReducers); //available reducers		
		// CPMM total costs
		double cpmm_costs =  (cpmm_shuffle1+cpmm_io1)/cpmm_nred1  //cpmm phase1
		                    +(cpmm_shuffle2+cpmm_io2)/cpmm_nred2; //cpmm phase2
		
		//final mmult method decision 
		if ( cpmm_costs < rmm_costs ) 
			return MMultMethod.CPMM;
		else 
			return MMultMethod.RMM;
	}

	@Override
	public SQLLops constructSQLLOPs() throws HopsException {
	
		if(this.get_sqllops() == null)
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
				this.set_sqllops(sqllop);
			}
			else
			{
				SQLLops sqllop = new SQLLops(this.get_name(),
										gen,
										hop1.constructSQLLOPs(),
										hop2.constructSQLLOPs(),
										this.get_valueType(), this.get_dataType());
	
				String sql = getSQLSelectCode(hop1, hop2);
			
				sqllop.set_sql(sql);
				sqllop.set_properties(getProperties(hop1, hop2));
			
				this.set_sqllops(sqllop);
			}
			this.set_visited(VISIT_STATUS.DONE);
		}
		return this.get_sqllops();
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
				+ hop1.get_sqllops().get_tableName() + " "
				+ Hop.HopsOpOp2String.get(innerOp) + " "
				+ hop1.get_sqllops().get_tableName() + ")");
		return prop;
	}
	
	private SQLSelectStatement getSQLSelect(Hop hop1, Hop hop2) throws HopsException
	{
		if(!(hop1.get_sqllops().get_dataType() == DataType.MATRIX && hop2.get_sqllops().get_dataType() == DataType.MATRIX))
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
		join.setTable1(new SQLTableReference(SQLLops.addQuotes(hop1.get_sqllops().get_tableName()), SQLLops.ALIAS_A));
		join.setTable2(new SQLTableReference(SQLLops.addQuotes(hop2.get_sqllops().get_tableName()), SQLLops.ALIAS_B));

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
		if(!(hop1.get_sqllops().get_dataType() == DataType.MATRIX && hop2.get_sqllops().get_dataType() == DataType.MATRIX))
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
						inner, hop1.get_sqllops().get_tableName(), hop2.get_sqllops().get_tableName());
			}
			//Special case for trace because it needs a special SELECT
			else if(this.outerOp == AggOp.TRACE)
			{
				sql = String.format(SQLLops.AGGTRACEOP, inner, hop1.get_sqllops().get_tableName(), join, hop2.get_sqllops().get_tableName());
			}
			//Should be handled before
			else if(this.outerOp == AggOp.SUM)
			{
				//sql = String.format(SQLLops.AGGSUMOP, inner, hop1.get_sqllops().get_tableName(), join, hop2.get_sqllops().get_tableName());
				//sql = getMatrixMultSQLString(inner, hop1.get_sqllops().get_tableName(), hop2.get_sqllops().get_tableName());
			}
			//Here the outerOp is just appended, it can only be min or max
			else
			{
				String outer = Hop.HopsAgg2String.get(this.outerOp);
				sql = String.format(SQLLops.AGGBINOP, outer, inner, 
					hop1.get_sqllops().get_tableName(), join, hop2.get_sqllops().get_tableName());
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
		
		boolean m_large = hop1.get_dim1() > SQLLops.HMATRIXSPLIT;
		boolean k_large = hop1.get_dim2() > SQLLops.VMATRIXSPLIT;
		boolean n_large = hop2.get_dim2() > SQLLops.HMATRIXSPLIT;
		
		String name = this.get_name() + "_" + this.getHopID();
		
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
			for(long s = SQLLops.HMATRIXSPLIT; s <= hop1.get_dim1(); s += SQLLops.HMATRIXSPLIT)
			{
				String where = SQLLops.ALIAS_A + ".row BETWEEN " + (s - SQLLops.HMATRIXSPLIT + 1) + " AND " + s;
				sb.append(String.format(SQLLops.SPLITAGGSUMOP, operation, i1, SQLLops.JOIN, i2, where));
				
				total = s;
				if(total < hop1.get_dim1())
					sb.append(" \r\nUNION ALL \r\n");
			}
			if(total < hop1.get_dim1())
			{
				String where = SQLLops.ALIAS_A + ".row BETWEEN " + (total + 1) + " AND " + hop1.get_dim1();
				sb.append(String.format(SQLLops.SPLITAGGSUMOP, operation, i1, SQLLops.JOIN, i2, where));
			}
			String sql = sb.toString();
			SQLLops lop = new SQLLops(name, flag, hop1.get_sqllops(), hop2.get_sqllops(), ValueType.DOUBLE, DataType.MATRIX);
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
			SQLLops h1 = getPart1SQLLop(operation, i1, i2, hop1.get_dim2() / 2, input1, input2);
			SQLLops h2 = getPart2SQLLop(operation, i1, i2, hop1.get_dim2() / 2, input1, input2);
			
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
			SQLLops lop = new SQLLops(name, flag, hop1.get_sqllops(), hop2.get_sqllops(), ValueType.DOUBLE, DataType.MATRIX);
			lop.set_sql(sql);
			return lop;
		}
	}
	
	private String getMatrixMultSQLString(String operation, String op1, String op2)
	{
		Hop hop1 = this.getInput().get(0);
		Hop hop2 = this.getInput().get(1);
		
		boolean m_large = hop1.get_dim1() > SQLLops.HMATRIXSPLIT;
		boolean k_large = hop1.get_dim2() > SQLLops.VMATRIXSPLIT;
		boolean n_large = hop2.get_dim2() > SQLLops.HMATRIXSPLIT;
		
		if(!SPLITLARGEMATRIXMULT || (!m_large && !k_large && !n_large))
			return String.format(SQLLops.AGGSUMOP, operation, op1, SQLLops.JOIN, op2);
		else
		{
			StringBuilder sb = new StringBuilder();
			//Split first matrix horizontally
			if(m_large)
			{
				long total = 0;
				for(long s = SQLLops.HMATRIXSPLIT; s <= hop1.get_dim1(); s += SQLLops.HMATRIXSPLIT)
				{
					String where = SQLLops.ALIAS_A + ".row BETWEEN " + (s - SQLLops.HMATRIXSPLIT + 1) + " AND " + s;
					sb.append(String.format(SQLLops.SPLITAGGSUMOP, operation, op1, SQLLops.JOIN, op2, where));
					
					total = s;
					if(total < hop1.get_dim1())
						sb.append(" \r\nUNION ALL \r\n");
				}
				if(total < hop1.get_dim1())
				{
					String where = SQLLops.ALIAS_A + ".row BETWEEN " + (total + 1) + " AND " + hop1.get_dim1();
					sb.append(String.format(SQLLops.SPLITAGGSUMOP, operation, op1, SQLLops.JOIN, op2, where));
				}
				return sb.toString();
			}
			//Split first matrix vertically and second matrix horizontally
			else if(k_large)
			{
				long middle = hop1.get_dim2() / 2;
				
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
			set_dim1(input1.get_dim1());
			set_dim2(input2.get_dim2());
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
				&& getInput().get(1) == that2.getInput().get(1));
	}
}
