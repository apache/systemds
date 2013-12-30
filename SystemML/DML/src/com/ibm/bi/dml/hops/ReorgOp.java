/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops;

import java.util.ArrayList;

import com.ibm.bi.dml.lops.Aggregate;
import com.ibm.bi.dml.lops.Group;
import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.lops.LopsException;
import com.ibm.bi.dml.lops.PartialAggregate;
import com.ibm.bi.dml.lops.Transform;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.sql.sqllops.SQLLopProperties;
import com.ibm.bi.dml.sql.sqllops.SQLLops;
import com.ibm.bi.dml.sql.sqllops.SQLLopProperties.AGGREGATIONTYPE;
import com.ibm.bi.dml.sql.sqllops.SQLLopProperties.JOINTYPE;
import com.ibm.bi.dml.sql.sqllops.SQLLops.GENERATES;

/**
 *  Reorg (cell) operation: aij
 * 		Properties: 
 * 			Symbol: ', diag, reshape
 * 			1 Operand
 * 	
 * 		Semantic: change indices (in mapper or reducer)
 * 
 * 
 *  NOTE MB: reshape integrated here because (1) ParameterizedBuiltinOp requires name-value pairs for params
 *  and (2) most importantly semantic of reshape is exactly a reorg op. 
 */

public class ReorgOp extends Hop 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private ReOrgOp op;

	private ReorgOp() {
		//default constructor for clone
	}
	
	public ReorgOp(String l, DataType dt, ValueType vt, ReOrgOp o, Hop inp) 
	{
		super(Kind.ReorgOp, l, dt, vt);
		op = o;
		getInput().add(0, inp);
		inp.getParent().add(this);
		
		//compute unknown dims and nnz
		refreshSizeInformation();
	}
	
	public ReorgOp(String l, DataType dt, ValueType vt, ReOrgOp o, ArrayList<Hop> inp) 
	{
		super(Kind.ReorgOp, l, dt, vt);
		op = o;
		
		for( int i=0; i<inp.size(); i++ ) {
			Hop in = inp.get(i);
			getInput().add(i, in);
			in.getParent().add(this);
		}
		
		//compute unknown dims and nnz
		refreshSizeInformation();
	}

	public ReOrgOp getOp()
	{
		return op;
	}
	
	@Override
	public String getOpString() {
		String s = new String("");
		s += "r(" + HopsTransf2String.get(op) + ")";
		return s;
	}

	@Override
	public Lop constructLops()
		throws HopsException, LopsException 
	{
		if (get_lops() == null) {
			
			ExecType et = optFindExecType();
			
			switch( op )
			{
				case DIAG_M2V: 
				{
					// Handle M2V case separately
					// partialAgg (diagM2V) - group - agg (+)
					
					if( et==ExecType.MR )
					{
						// TODO: this code must be revisited once the support for matrix indexing is implemented. 
						
						PartialAggregate transform1 = new PartialAggregate(
								getInput().get(0).constructLops(),
								Aggregate.OperationTypes.DiagM2V,
								HopsDirection2Lops.get(Direction.Col),
								get_dataType(), get_valueType(), et);
						transform1.getOutputParameters().setDimensions(get_dim1(),
								get_dim2(), get_rows_in_block(),
								get_cols_in_block(), getNnz());
						transform1.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());

						Group group1 = new Group(
								transform1, Group.OperationTypes.Sort,
								get_dataType(), get_valueType());
						group1.getOutputParameters().setDimensions(get_dim1(),
								get_dim2(), get_rows_in_block(),
								get_cols_in_block(), getNnz());						
						group1.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
	
						Aggregate agg1 = new Aggregate(
								group1, HopsAgg2Lops.get(AggOp.SUM),
								get_dataType(), get_valueType(), ExecType.MR);
						agg1.getOutputParameters().setDimensions(get_dim1(),
								get_dim2(), get_rows_in_block(),
								get_cols_in_block(), getNnz());	
						agg1.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
						
						// kahanSum setup is not used for Diag operations. They are
						// treated as special case in the run time
						agg1.setupCorrectionLocation(transform1.getCorrectionLocaion());
	
						set_lops(agg1);
					}
					else //CP
					{
						PartialAggregate transform1 = new PartialAggregate(
								getInput().get(0).constructLops(),
								Aggregate.OperationTypes.DiagM2V,
								HopsDirection2Lops.get(Direction.Col),
								get_dataType(), get_valueType(), et);
						transform1.getOutputParameters().setDimensions(get_dim1(),
								get_dim2(), get_rows_in_block(),
								get_cols_in_block(), getNnz());
						transform1.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());

						set_lops(transform1);
					}	
					break;
				}
				case TRANSPOSE:
				case DIAG_V2M:
				{
					Transform transform1 = new Transform(
							getInput().get(0).constructLops(), HopsTransf2Lops
									.get(op), get_dataType(), get_valueType(), et);
					transform1.getOutputParameters().setDimensions(get_dim1(),
							get_dim2(), get_rows_in_block(), get_cols_in_block(), getNnz());	
					transform1.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
					
					set_lops(transform1);
					break;
				}
				case RESHAPE:
				{
					if( et==ExecType.MR )
					{
						Transform transform1 = new Transform(
								getInput().get(0).constructLops(), HopsTransf2Lops.get(op), 
								get_dataType(), get_valueType(), et);
						transform1.getOutputParameters().setDimensions(get_dim1(),
								get_dim2(), get_rows_in_block(), get_cols_in_block(), getNnz());	
						transform1.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
						for( int i=1; i<=3; i++ ) //rows, cols, byrow
						{
							Lop ltmp = getInput().get(i).constructLops();
							transform1.addInput(ltmp);
							ltmp.addOutput(transform1);
						}
						transform1.setLevel(); //force order of added lops
						
						Group group1 = new Group(
								transform1, Group.OperationTypes.Sort, DataType.MATRIX,
								get_valueType());
						group1.getOutputParameters().setDimensions(get_dim1(),
								get_dim2(), get_rows_in_block(), get_cols_in_block(), getNnz());
						group1.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
		
						Aggregate agg1 = new Aggregate(
								group1, Aggregate.OperationTypes.Sum, DataType.MATRIX,
								get_valueType(), et);
						agg1.getOutputParameters().setDimensions(get_dim1(),
								get_dim2(), get_rows_in_block(), get_cols_in_block(), getNnz());		
						agg1.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
						
						set_lops(agg1);
					}
					else //CP
					{
						Transform transform1 = new Transform(
								getInput().get(0).constructLops(), HopsTransf2Lops.get(op), 
								get_dataType(), get_valueType(), et);
						transform1.getOutputParameters().setDimensions(get_dim1(),
								get_dim2(), get_rows_in_block(), get_cols_in_block(), getNnz());	
						transform1.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
						for( int i=1; i<=3; i++ ) //rows, cols, byrow
						{
							Lop ltmp = getInput().get(i).constructLops();
							transform1.addInput(ltmp);
							ltmp.addOutput(transform1);
						}
						transform1.setLevel(); //force order of added lops
						
						set_lops(transform1);
					}
				}
			}
		}
		
		return get_lops();
	}

	@Override
	public SQLLops constructSQLLOPs() throws HopsException {
		if (this.get_sqllops() == null) {
			if (this.getInput().size() != 1)
				throw new HopsException(this.printErrorLocation() + "An unary hop must have only one input \n");

			// Check whether this is going to be an Insert or With
			GENERATES gen = determineGeneratesFlag();

			Hop input = this.getInput().get(0);

			SQLLops sqllop = new SQLLops(this.get_name(),
										gen,
										input.constructSQLLOPs(),
										this.get_valueType(),
										this.get_dataType());

			String sql = null;
			if (this.op == ReOrgOp.TRANSPOSE) {
				sql = String.format(SQLLops.TRANSPOSEOP, input.get_sqllops().get_tableName());
			} else if (op == ReOrgOp.DIAG_M2V) {
				sql = String.format(SQLLops.DIAG_M2VOP, input.get_sqllops().get_tableName());
			} else if (op == ReOrgOp.DIAG_V2M) {
				sql = String.format(SQLLops.DIAG_V2M, input.get_sqllops().get_tableName());
			}
			//TODO reshape
			
			sqllop.set_properties(getProperties(input));
			sqllop.set_sql(sql);

			this.set_sqllops(sqllop);
		}
		return this.get_sqllops();
	}
	
	private SQLLopProperties getProperties(Hop input)
	{
		SQLLopProperties prop = new SQLLopProperties();
		prop.setJoinType(JOINTYPE.NONE);
		prop.setAggType(AGGREGATIONTYPE.NONE);
		prop.setOpString(HopsTransf2String.get(op) + "(" + input.get_sqllops().get_tableName() + ")");
		return prop;
	}
		
	@Override
	protected double computeOutputMemEstimate( long dim1, long dim2, long nnz )
	{		
		//no dedicated mem estimation per op type, because always propagated via refreshSizeInformation
		double sparsity = OptimizerUtils.getSparsity(dim1, dim2, nnz);
		return OptimizerUtils.estimateSizeExactSparsity(dim1, dim2, sparsity);
	}
	
	@Override
	protected double computeIntermediateMemEstimate( long dim1, long dim2, long nnz )
	{
		return 0;
	}
	
	@Override
	protected long[] inferOutputCharacteristics( MemoTable memo )
	{
		long[] ret = null;
	
		Hop input = getInput().get(0);
		MatrixCharacteristics mc = memo.getAllInputStats(input);
			
		switch(op) 
		{
			case TRANSPOSE:
			{
				// input is a [k1,k2] matrix and output is a [k2,k1] matrix
				// #nnz in output is exactly the same as in input
				if( mc.dimsKnown() )
					ret = new long[]{ mc.get_cols(), mc.get_rows(), mc.getNonZeros() };
				break;
			}	
			case DIAG_V2M:
			{
				// input is a [1,k] or [k,1] matrix, and output is [k,k] matrix
				// #nnz in output is in the worst case k => sparsity = 1/k
				
				//the following approach leads to problems if dim1 unknown, but dim2 known to be 1
				//long k = (mc.get_rows() > 1 ? mc.get_rows() : mc.get_cols()); 
				long k = mc.get_rows(); 
				if( k > 0 )
					ret = new long[]{k, k, ((mc.getNonZeros()>0) ? mc.getNonZeros() : k)};
				break;
			}	
			case DIAG_M2V:
			{
				// input is [k,k] matrix and output is [k,1] matrix
				// #nnz in the output is likely to be k (a dense matrix)		
				long k = mc.get_rows();
				if( k > 0 )
					ret = new long[]{k, 1, ((mc.getNonZeros()>0) ? Math.min(k,mc.getNonZeros()) : k) };
				break;		
			}
			case RESHAPE:
			{
				// input is a [k1,k2] matrix and output is a [k3,k4] matrix with k1*k2=k3*k4
				// #nnz in output is exactly the same as in input		
				if( mc.dimsKnown() ) {
					if( _dim1 > 0  )
						ret = new long[]{ _dim1, mc.get_rows()*mc.get_cols()/_dim1, mc.nonZero};
					else if( _dim2 > 0 )	 
						ret = new long[]{ mc.get_rows()*mc.get_cols()/_dim2, _dim2, mc.nonZero};
				}
				break;
			}
		}	
		
		return ret;
	}
	

	@Override
	public boolean allowsAllExecTypes()
	{
		return true;
	}
	
	@Override
	protected ExecType optFindExecType() throws HopsException {
		
		checkAndSetForcedPlatform();
	
		if( _etypeForced != null ) 			
			_etype = _etypeForced;
		else 
		{	
			if ( OptimizerUtils.isMemoryBasedOptLevel() ) {
				_etype = findExecTypeByMemEstimate();
			}
			// Choose CP, if the input dimensions are below threshold or if the input is a vector
			else if ( getInput().get(0).areDimsBelowThreshold() || getInput().get(0).isVector() )
				_etype = ExecType.CP;
			else 
				_etype = ExecType.MR;
			
			//mark for recompile (forever)
			if( OptimizerUtils.ALLOW_DYN_RECOMPILATION && !dimsKnown(true) && _etype==ExecType.MR )
				setRequiresRecompile();
		}
		
		return _etype;
	}
	
	@Override
	public void refreshSizeInformation()
	{
		Hop input1 = getInput().get(0);
		
		switch(op) 
		{
			case TRANSPOSE:
			{
				// input is a [k1,k2] matrix and output is a [k2,k1] matrix
				// #nnz in output is exactly the same as in input
				set_dim1(input1.get_dim2());
				set_dim2(input1.get_dim1());
				setNnz(input1.getNnz());
				break;
			}	
			case DIAG_V2M:
			{
				// input is a [1,k] or [k,1] matrix, and output is [k,k] matrix
				// #nnz in output is in the worst case k => sparsity = 1/k
				
				//the following approach leads to problems if dim1 unknown, but dim2 known to be 1
				//long k = (input1.get_dim1() > 1 ? input1.get_dim1() : input1.get_dim2()); 
				long k = input1.get_dim1(); 
				set_dim1(k);
				set_dim2(k);
				setNnz( (input1.getNnz()>0) ? input1.getNnz() : k );
				break;
			}	
			case DIAG_M2V:
			{
				// input is [k,k] matrix and output is [k,1] matrix
				// #nnz in the output is likely to be k (a dense matrix)		
				long k = input1.get_dim1();
				set_dim1(k);
				set_dim2(1);	
				setNnz( (input1.getNnz()>0) ? Math.min(k,input1.getNnz()) : k );
				break;		
			}
			case RESHAPE:
			{
				// input is a [k1,k2] matrix and output is a [k3,k4] matrix with k1*k2=k3*k4
				// #nnz in output is exactly the same as in input		
				Hop input2 = getInput().get(1); //rows 
				Hop input3 = getInput().get(2); //cols 
				refreshRowsParameterInformation(input2); //refresh rows
 				refreshColsParameterInformation(input3); //refresh cols
 				setNnz(input1.getNnz());
 				if( !dimsKnown() &&input1.dimsKnown() ) { //reshape allows to infer dims, if input and 1 dim known
	 				if(_dim1 > 0) 
						_dim2 = (input1._dim1*input1._dim2)/_dim1;
					else if(_dim2 > 0)	
						_dim1 = (input1._dim1*input1._dim2)/_dim2; 
 				}
				break;
			}
		}	
	}
	
	@Override
	public Object clone() throws CloneNotSupportedException 
	{
		ReorgOp ret = new ReorgOp();	
		
		//copy generic attributes
		ret.clone(this, false);
		
		//copy specific attributes
		ret.op = op;
		
		return ret;
	}
	
	@Override
	public boolean compare( Hop that )
	{
		if( that._kind!=Kind.ReorgOp )
			return false;
		
		ReorgOp that2 = (ReorgOp)that;		
		return (   op == that2.op
				&& getInput().get(0) == that2.getInput().get(0));
	}
	
	
	@Override
	public void printMe() throws HopsException 
	{
		if (LOG.isDebugEnabled()){
			if (get_visited() != VISIT_STATUS.DONE) {
				super.printMe();
				LOG.debug("  Operation: " + op);
				for (Hop h : getInput()) {
					h.printMe();
				}
			}
			set_visited(VISIT_STATUS.DONE);
		}
	}
	
}