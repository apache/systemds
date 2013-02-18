package com.ibm.bi.dml.hops;

import com.ibm.bi.dml.hops.OptimizerUtils.OptimizationType;
import com.ibm.bi.dml.lops.Binary;
import com.ibm.bi.dml.lops.Group;
import com.ibm.bi.dml.lops.LeftIndex;
import com.ibm.bi.dml.lops.Lops;
import com.ibm.bi.dml.lops.RangeBasedReIndex;
import com.ibm.bi.dml.lops.UnaryCP;
import com.ibm.bi.dml.lops.ZeroOut;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.UnaryCP.OperationTypes;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.sql.sqllops.SQLLops;
import com.ibm.bi.dml.utils.HopsException;

public class LeftIndexingOp  extends Hops {

	public static String OPSTRING = "LeftIndexing";
	
	public LeftIndexingOp(String l, DataType dt, ValueType vt, Hops inpMatrixLeft, Hops inpMatrixRight, Hops inpRowL, Hops inpRowU, Hops inpColL, Hops inpColU) {
		super(Kind.Indexing, l, dt, vt);

		getInput().add(0, inpMatrixLeft);
		getInput().add(1, inpMatrixRight);
		getInput().add(2, inpRowL);
		getInput().add(3, inpRowU);
		getInput().add(4, inpColL);
		getInput().add(5, inpColU);
		
		// create hops if one of them is null
		inpMatrixLeft.getParent().add(this);
		inpMatrixRight.getParent().add(this);
		inpRowL.getParent().add(this);
		inpRowU.getParent().add(this);
		inpColL.getParent().add(this);
		inpColU.getParent().add(this);
	}

	public Lops constructLops()
			throws HopsException {
		if (get_lops() == null) {
			try {
				ExecType et = optFindExecType();
				if(et == ExecType.MR) {
					
					//the right matrix is reindexed
					Lops top=getInput().get(2).constructLops();
					Lops bottom=getInput().get(3).constructLops();
					Lops left=getInput().get(4).constructLops();
					Lops right=getInput().get(5).constructLops();
					/*
					//need to creat new lops for converting the index ranges
					//original range is (a, b) --> (c, d)
					//newa=2-a, newb=2-b
					Lops two=new Data(null,	Data.OperationTypes.READ, null, "2", Expression.DataType.SCALAR, Expression.ValueType.INT, false);
					Lops newTop=new Binary(two, top, HopsOpOp2LopsB.get(Hops.OpOp2.MINUS), Expression.DataType.SCALAR, Expression.ValueType.INT, et);
					Lops newLeft=new Binary(two, left, HopsOpOp2LopsB.get(Hops.OpOp2.MINUS), Expression.DataType.SCALAR, Expression.ValueType.INT, et);
					//newc=leftmatrix.row-a+1, newd=leftmatrix.row
					*/
					//right hand matrix
					Lops nrow=new UnaryCP(getInput().get(0).constructLops(), 
									OperationTypes.NROW, DataType.SCALAR, ValueType.INT);
					Lops ncol=new UnaryCP(getInput().get(0).constructLops(), 
											OperationTypes.NCOL, DataType.SCALAR, ValueType.INT);
					RangeBasedReIndex reindex = new RangeBasedReIndex(
							getInput().get(1).constructLops(), top, bottom, 
							left, right, nrow, ncol,
							get_dataType(), get_valueType(), et, true);
					
					reindex.getOutputParameters().setDimensions(getInput().get(0).get_dim1(), getInput().get(0).get_dim2(), 
							get_rows_in_block(), get_cols_in_block(), getNnz());
					
					reindex.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
					
					Group group1 = new Group(
							reindex, Group.OperationTypes.Sort, DataType.MATRIX,
							get_valueType());
					group1.getOutputParameters().setDimensions(getInput().get(0).get_dim1(), getInput().get(0).get_dim2(), 
							get_rows_in_block(), get_cols_in_block(), getNnz());
					
					group1.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
	
					//the left matrix is zeroed out
					ZeroOut zeroout = new ZeroOut(
							getInput().get(0).constructLops(), top, bottom,
							left, right, getInput().get(0).get_dim1(), getInput().get(0).get_dim2(),
							get_dataType(), get_valueType(), et);
	
					zeroout.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
					
					zeroout.getOutputParameters().setDimensions(getInput().get(0).get_dim1(), getInput().get(0).get_dim2(), 
							get_rows_in_block(), get_cols_in_block(), getNnz());
					Group group2 = new Group(
							zeroout, Group.OperationTypes.Sort, DataType.MATRIX,
							get_valueType());
					group2.getOutputParameters().setDimensions(getInput().get(0).get_dim1(), getInput().get(0).get_dim2(), 
							get_rows_in_block(), get_cols_in_block(), getNnz());
					
					group2.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
					
					Binary binary = new Binary(group1, group2, HopsOpOp2LopsB.get(Hops.OpOp2.PLUS),
							get_dataType(), get_valueType(), et);
					
					binary.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
					
					binary.getOutputParameters().setDimensions(getInput().get(0).get_dim1(), getInput().get(0).get_dim2(), 
							get_rows_in_block(), get_cols_in_block(), getNnz());
	
					set_lops(binary);
				}
				else {
					LeftIndex left = new LeftIndex(
							getInput().get(0).constructLops(), getInput().get(1).constructLops(), getInput().get(2).constructLops(), 
							getInput().get(3).constructLops(), getInput().get(4).constructLops(), getInput().get(5).constructLops(), 
							get_dataType(), get_valueType(), et);
					left.getOutputParameters().setDimensions(get_dim1(), get_dim2(), get_rows_in_block(), get_cols_in_block(), getNnz());
					left.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
					set_lops(left);
				}
			} catch (Exception e) {
				throw new HopsException(this.printErrorLocation() + "In LeftIndexingOp Hop, error in constructing Lops " , e);
			}

		}
		return get_lops();
	}

	@Override
	public String getOpString() {
		String s = new String("");
		s += OPSTRING;
		return s;
	}

	public void printMe() throws HopsException {
		if (get_visited() != VISIT_STATUS.DONE) {
			super.printMe();
			for (Hops h : getInput()) {
				h.printMe();
			}
			;
		}
		set_visited(VISIT_STATUS.DONE);
	}

	public SQLLops constructSQLLOPs() throws HopsException {
		throw new HopsException(this.printErrorLocation() + "constructSQLLOPs should not be called for LeftIndexingOp \n");
	}
	
	@Override
	public boolean allowsAllExecTypes()
	{
		return false;
	}
	
	@Override
	public double computeMemEstimate() {
		
		//1) output mem estimate
		if ( dimsKnown() ) {
			// The dimensions of the left indexing output is same as that of the first input i.e., getInput().get(0)
			// However, the sparsity might change -- TODO: we can not handle the change in sparsity, for now
			_outputMemEstimate = OptimizerUtils.estimateSize(_dim1, _dim2, getInput().get(0).getSparsity());
		}
		else {
			_outputMemEstimate = OptimizerUtils.DEFAULT_SIZE;
		}
		
		//2) operation mem estimate
		if( !getInput().get(1).dimsKnown() ) {
			//use worst-case memory estimate for second input (it cannot be larger than overall matrix)
			_memEstimate = 2 * _outputMemEstimate;
		}
		else
			_memEstimate = getInputOutputSize();
		
		return _memEstimate;
	}
	
	@Override
	protected ExecType optFindExecType() throws HopsException {
		
		checkAndSetForcedPlatform();
		
		if( _etypeForced != null ) 			
			_etype = _etypeForced;
		else 
		{	
			//mark for recompile (forever)
			if( OptimizerUtils.ALLOW_DYN_RECOMPILATION && !dimsKnown() )
				setRequiresRecompile();
			
			if ( OptimizerUtils.getOptType() == OptimizationType.MEMORY_BASED ) {
				_etype = findExecTypeByMemEstimate();
				checkAndModifyRecompilationStatus();
			}
			else if ( getInput().get(0).areDimsBelowThreshold() )
				_etype = ExecType.CP;
			else 
				_etype = ExecType.MR;
		}
		return _etype;
	}

	@Override
	public void refreshSizeInformation()
	{
		Hops input1 = getInput().get(0);		
		set_dim1( input1.get_dim1() );
		set_dim2( input1.get_dim2() );
	}
	
	/**
	 * 
	 */
	private void checkAndModifyRecompilationStatus()
	{
		// disable recompile for LIX and scond input matrix (under certain conditions)
		// if worst-case estimate (2 * original matrix size) was enough to already send it to CP 		
		
		if( _etype == ExecType.CP )
		{
			_requiresRecompile = false;
			
			Hops rInput = getInput().get(1);
			if( !rInput.dimsKnown() && rInput instanceof DataOp  )
				rInput._requiresRecompile=false;
		}
	}
	
}
