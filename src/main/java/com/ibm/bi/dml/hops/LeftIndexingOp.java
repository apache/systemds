/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package com.ibm.bi.dml.hops;

import com.ibm.bi.dml.lops.Binary;
import com.ibm.bi.dml.lops.Group;
import com.ibm.bi.dml.lops.LeftIndex;
import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.lops.LopsException;
import com.ibm.bi.dml.lops.RangeBasedReIndex;
import com.ibm.bi.dml.lops.UnaryCP;
import com.ibm.bi.dml.lops.ZeroOut;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.UnaryCP.OperationTypes;
import com.ibm.bi.dml.parser.DMLTranslator;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;

public class LeftIndexingOp  extends Hop 
{
	
	public static LeftIndexingMethod FORCED_LEFT_INDEXING = null;
	
	public enum LeftIndexingMethod { 
		SP_GLEFTINDEX, // general case
		SP_MLEFTINDEX //map-only left index where we broadcast right hand side matrix
	}
	
	public static String OPSTRING = "lix"; //"LeftIndexing";
	
	private boolean _rowLowerEqualsUpper = false, _colLowerEqualsUpper = false;
	
	public boolean getRowLowerEqualsUpper(){
		return _rowLowerEqualsUpper;
	}
	
	public boolean getColLowerEqualsUpper() {
		return _colLowerEqualsUpper;
	}
	
	public void setRowLowerEqualsUpper(boolean passed){
		_rowLowerEqualsUpper  = passed;
	}
	
	public void setColLowerEqualsUpper(boolean passed) {
		_colLowerEqualsUpper = passed;
	}
	
	private LeftIndexingMethod getOptMethodLeftIndexingMethod( long m1_dim1, long m1_dim2, long m2_dim1, long m2_dim2, long m1_rpb, long m1_cpb ) {
		if(FORCED_LEFT_INDEXING != null) {
			return FORCED_LEFT_INDEXING;
		}
		// Broadcast based left index behaves like a map binary op in worst case where we fit entire rhs broadcast in memory
		// and only one block from lhs and output
		double size = OptimizerUtils.estimateSize(m2_dim1, m2_dim2);
		if( OptimizerUtils.checkSparkBroadcastMemoryBudget(size) ) {
			return LeftIndexingMethod.SP_MLEFTINDEX;
		}
		return LeftIndexingMethod.SP_GLEFTINDEX;
	}
	
	private LeftIndexingOp() {
		//default constructor for clone
	}
	
	public LeftIndexingOp(String l, DataType dt, ValueType vt, Hop inpMatrixLeft, Hop inpMatrixRight, Hop inpRowL, Hop inpRowU, Hop inpColL, Hop inpColU, boolean passedRowsLEU, boolean passedColsLEU) {
		super(l, dt, vt);

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
		
		// set information whether left indexing operation involves row (n x 1) or column (1 x m) matrix
		setRowLowerEqualsUpper(passedRowsLEU);
		setColLowerEqualsUpper(passedColsLEU);
	}

	
	@Override
	public Lop constructLops()
		throws HopsException, LopsException 
	{			
		//return already created lops
		if( getLops() != null )
			return getLops();

		try 
		{
			ExecType et = optFindExecType();
			
			if(et == ExecType.MR) 
			{	
				//the right matrix is reindexed
				Lop top=getInput().get(2).constructLops();
				Lop bottom=getInput().get(3).constructLops();
				Lop left=getInput().get(4).constructLops();
				Lop right=getInput().get(5).constructLops();
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
				Lop nrow=new UnaryCP(getInput().get(0).constructLops(), 
								OperationTypes.NROW, DataType.SCALAR, ValueType.INT);
				Lop ncol=new UnaryCP(getInput().get(0).constructLops(), 
										OperationTypes.NCOL, DataType.SCALAR, ValueType.INT);
				
				Lop rightInput = null;
				if (isRightHandSideScalar()) {
					//insert cast to matrix if necessary (for reuse MR runtime)
					rightInput = new UnaryCP(getInput().get(1).constructLops(),
							                 OperationTypes.CAST_AS_MATRIX, 
							                 DataType.MATRIX, ValueType.DOUBLE);
					rightInput.getOutputParameters().setDimensions( (long)1, (long)1,
																	(long)DMLTranslator.DMLBlockSize, 
							                                        (long)DMLTranslator.DMLBlockSize,
							                                        (long)-1);
				} 
				else 
					rightInput = getInput().get(1).constructLops();

				
				RangeBasedReIndex reindex = new RangeBasedReIndex(
						rightInput, top, bottom, 
						left, right, nrow, ncol,
						getDataType(), getValueType(), et, true);
				
				reindex.getOutputParameters().setDimensions(getInput().get(0).getDim1(), getInput().get(0).getDim2(), 
						getRowsInBlock(), getColsInBlock(), getNnz());
				setLineNumbers(reindex);
				
				Group group1 = new Group(
						reindex, Group.OperationTypes.Sort, DataType.MATRIX,
						getValueType());
				group1.getOutputParameters().setDimensions(getInput().get(0).getDim1(), getInput().get(0).getDim2(), 
						getRowsInBlock(), getColsInBlock(), getNnz());
				setLineNumbers(group1);
				
				//the left matrix is zeroed out
				ZeroOut zeroout = new ZeroOut(
						getInput().get(0).constructLops(), top, bottom,
						left, right, getInput().get(0).getDim1(), getInput().get(0).getDim2(),
						getDataType(), getValueType(), et);
				zeroout.getOutputParameters().setDimensions(getInput().get(0).getDim1(), getInput().get(0).getDim2(), 
						getRowsInBlock(), getColsInBlock(), getNnz());
				setLineNumbers(zeroout);
				
				Group group2 = new Group(
						zeroout, Group.OperationTypes.Sort, DataType.MATRIX,
						getValueType());
				group2.getOutputParameters().setDimensions(getInput().get(0).getDim1(), getInput().get(0).getDim2(), 
						getRowsInBlock(), getColsInBlock(), getNnz());
				setLineNumbers(group2);
				
				Binary binary = new Binary(group1, group2, HopsOpOp2LopsB.get(Hop.OpOp2.PLUS),
						getDataType(), getValueType(), et);				
				binary.getOutputParameters().setDimensions(getInput().get(0).getDim1(), getInput().get(0).getDim2(), 
						getRowsInBlock(), getColsInBlock(), getNnz());
				setLineNumbers(binary);
				
				setLops(binary);
			}
			else if(et == ExecType.SPARK)  {				
				boolean isBroadcast = false;
				Hop left = getInput().get(0);
				Hop right = getInput().get(0);
				long m1_dim1 = left.getDim1();
				long m1_dim2 = left.getDim2();		
				long m2_dim1 = right.getDim1();
				long m2_dim2 = right.getDim2();
				long brlen = left.getRowsInBlock();
				long bclen = left.getColsInBlock();
				LeftIndexingMethod method = getOptMethodLeftIndexingMethod(m1_dim1, m1_dim2, m2_dim1, m2_dim2, brlen, bclen);
				if(method == LeftIndexingMethod.SP_MLEFTINDEX) {
					isBroadcast = true;
				}
				
				LeftIndex leftIndexLop = new LeftIndex(
						getInput().get(0).constructLops(), getInput().get(1).constructLops(), getInput().get(2).constructLops(), 
						getInput().get(3).constructLops(), getInput().get(4).constructLops(), getInput().get(5).constructLops(), 
						getDataType(), getValueType(), et, isBroadcast);
				
				setOutputDimensions(leftIndexLop);
				setLineNumbers(leftIndexLop);
				setLops(leftIndexLop);
			}
			else 
			{
				LeftIndex left = new LeftIndex(
						getInput().get(0).constructLops(), getInput().get(1).constructLops(), getInput().get(2).constructLops(), 
						getInput().get(3).constructLops(), getInput().get(4).constructLops(), getInput().get(5).constructLops(), 
						getDataType(), getValueType(), et);
				
				setOutputDimensions(left);
				setLineNumbers(left);
				setLops(left);
			}
		} 
		catch (Exception e) {
			throw new HopsException(this.printErrorLocation() + "In LeftIndexingOp Hop, error in constructing Lops " , e);
		}

		//add reblock/checkpoint lops if necessary
		constructAndSetLopsDataFlowProperties();
		
		return getLops();
	}
	
	/**
	 * @return true if the right hand side of the indexing operation is a
	 *         literal.
	 */
	private boolean isRightHandSideScalar() {
		Hop rightHandSide = getInput().get(1);
		return (rightHandSide.getDataType() == DataType.SCALAR);
	}
	
	@Override
	public String getOpString() {
		String s = new String("");
		s += OPSTRING;
		return s;
	}

	public void printMe() throws HopsException {
		if (getVisited() != VisitStatus.DONE) {
			super.printMe();
			for (Hop h : getInput()) {
				h.printMe();
			}
			;
		}
		setVisited(VisitStatus.DONE);
	}

	@Override
	public boolean allowsAllExecTypes()
	{
		return false;
	}

	@Override
	public void computeMemEstimate( MemoTable memo ) 
	{
		//overwrites default hops behavior
		super.computeMemEstimate(memo);	
		
		//changed final estimate (infer and use input size)
		Hop rhM = getInput().get(1);
		MatrixCharacteristics mcRhM = memo.getAllInputStats(rhM);
		//TODO also use worstcase estimate for output
		if( dimsKnown() && !(rhM.dimsKnown()||mcRhM.dimsKnown()) ) 
		{ 
			// unless second input is single cell / row vector / column vector
			// use worst-case memory estimate for second input (it cannot be larger than overall matrix)
			double subSize = -1;	
			if( _rowLowerEqualsUpper && _colLowerEqualsUpper )
				subSize = OptimizerUtils.estimateSize(1, 1);	
			else if( _rowLowerEqualsUpper )
				subSize = OptimizerUtils.estimateSize(1, _dim2);
			else if( _colLowerEqualsUpper )
				subSize = OptimizerUtils.estimateSize(_dim1, 1);
			else 
				subSize = _outputMemEstimate; //worstcase

			_memEstimate = getInputSize(0) //original matrix (left)
			               + subSize // new submatrix (right)
			               + _outputMemEstimate; //output size (output)
		}
		else if ( dimsKnown() && _nnz<0 &&
				  _memEstimate>=OptimizerUtils.DEFAULT_SIZE)
		{
			//try a last attempt to infer a reasonable estimate wrt output sparsity
			//(this is important for indexing sparse matrices into empty matrices).
			MatrixCharacteristics mcM1 = memo.getAllInputStats(getInput().get(0));
			MatrixCharacteristics mcM2 = memo.getAllInputStats(getInput().get(1));
			if( mcM1.getNonZeros()>=0 && mcM2.getNonZeros()>=0  ) {
				long lnnz = mcM1.getNonZeros() + mcM2.getNonZeros();
				_outputMemEstimate = computeOutputMemEstimate( _dim1, _dim2, lnnz );
				_memEstimate = getInputSize(0) //original matrix (left)
			                 + getInputSize(1) // new submatrix (right)
			                 + _outputMemEstimate; //output size (output)
			}
		}
	}
	
	@Override
	protected double computeOutputMemEstimate( long dim1, long dim2, long nnz )
	{	
		double sparsity = 1.0;
		if( nnz < 0 ) //check for exactly known nnz
		{
			Hop input1 = getInput().get(0);
			Hop input2 = getInput().get(1);
			if( input1.dimsKnown() ) {
				sparsity = OptimizerUtils.getLeftIndexingSparsity(
						input1.getDim1(), input1.getDim2(), input1.getNnz(), 
						input2.getDim1(), input2.getDim2(), input2.getNnz());
			}
		}
		else
		{
			sparsity = OptimizerUtils.getSparsity(dim1, dim2, nnz);
		}
		
		// The dimensions of the left indexing output is same as that of the first input i.e., getInput().get(0)
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
	
		Hop input1 = getInput().get(0); //original matrix
		Hop input2 = getInput().get(1); //right matrix		
		MatrixCharacteristics mc1 = memo.getAllInputStats(input1);
		MatrixCharacteristics mc2 = memo.getAllInputStats(input2);
		
		if( mc1.dimsKnown() ) {
			double sparsity = OptimizerUtils.getLeftIndexingSparsity(
					mc1.getRows(), mc1.getCols(), mc1.getNonZeros(), 
					mc2.getRows(), mc2.getCols(), mc2.getNonZeros());
			long lnnz = (long)(sparsity * mc1.getRows() * mc1.getCols());
			        
			ret = new long[]{mc1.getRows(), mc1.getCols(), lnnz};
		}
		
		return ret;
	}
	
	
	@Override
	protected ExecType optFindExecType() throws HopsException {
		
		checkAndSetForcedPlatform();
		
		ExecType REMOTE = OptimizerUtils.isSparkExecutionMode() ? ExecType.SPARK : ExecType.MR;
		
		if( _etypeForced != null ) 			
		{
			_etype = _etypeForced;
		}
		else 
		{	
			if ( OptimizerUtils.isMemoryBasedOptLevel() ) {
				_etype = findExecTypeByMemEstimate();
				checkAndModifyRecompilationStatus();
			}
			else if ( getInput().get(0).areDimsBelowThreshold() )
			{
				_etype = ExecType.CP;
			}
			else 
			{
				_etype = REMOTE;
			}
			
			//check for valid CP dimensions and matrix size
			checkAndSetInvalidCPDimsAndSize();
		}
		
		//mark for recompile (forever)
		if( OptimizerUtils.ALLOW_DYN_RECOMPILATION && !dimsKnown(true) && _etype==REMOTE )
			setRequiresRecompile();
	
		return _etype;
	}

	@Override
	public void refreshSizeInformation()
	{
		Hop input1 = getInput().get(0);	//original matrix	
		Hop input2 = getInput().get(1); //rhs matrix
		
		//refresh output dimensions based on original matrix
		setDim1( input1.getDim1() );
		setDim2( input1.getDim2() );
		
		//refresh output nnz if exactly known; otherwise later inference
		if( input1.getNnz() == 0 )  {
			if( input2.getDataType()==DataType.SCALAR )
				setNnz(1);
			else 
				setNnz(input2.getNnz());
		}
		else
			setNnz(-1);
	}
	
	/**
	 * 
	 */
	private void checkAndModifyRecompilationStatus()
	{
		// disable recompile for LIX and second input matrix (under certain conditions)
		// if worst-case estimate (2 * original matrix size) was enough to already send it to CP 		
		
		if( _etype == ExecType.CP )
		{
			_requiresRecompile = false;
			
			Hop rInput = getInput().get(1);
			if( (!rInput.dimsKnown()) && rInput instanceof DataOp  )
			{
				//disable recompile for this dataop (we cannot set requiresRecompile directly 
				//because we use a top-down traversal for creating lops, hence it would be overwritten)
				
				((DataOp)rInput).disableRecompileRead();
			}
		}
	}
	
	@Override
	public Object clone() throws CloneNotSupportedException 
	{
		LeftIndexingOp ret = new LeftIndexingOp();	
		
		//copy generic attributes
		ret.clone(this, false);
		
		//copy specific attributes
		
		return ret;
	}
	
	@Override
	public boolean compare( Hop that )
	{
		if(    !(that instanceof LeftIndexingOp) 
			|| getInput().size() != that.getInput().size() )
		{
			return false;
		}
		
		return (  getInput().get(0) == that.getInput().get(0)
				&& getInput().get(1) == that.getInput().get(1)
				&& getInput().get(2) == that.getInput().get(2)
				&& getInput().get(3) == that.getInput().get(3)
				&& getInput().get(4) == that.getInput().get(4)
				&& getInput().get(5) == that.getInput().get(5));
	}

}
