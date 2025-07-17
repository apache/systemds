/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysds.hops;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ReOrgOp;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.conf.CompilerConfig;
import org.apache.sysds.hops.rewrite.HopRewriteUtils;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.lops.Transform;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;

import java.util.List;

/**
 *  Reorg (cell) operation: aij
 * 		Properties: 
 * 			Symbol: ', rdiag, rshape, rsort
 * 			1 Operand (except sort and reshape take additional arguments)
 * 	
 * 		Semantic: change indices (in mapper or reducer)
 * 
 * 
 *  NOTE MB: reshape integrated here because (1) ParameterizedBuiltinOp requires name-value pairs for params
 *  and (2) most importantly semantic of reshape is exactly a reorg op. 
 */

public class ReorgOp extends MultiThreadedHop
{
	public static boolean FORCE_DIST_SORT_INDEXES = false;
	
	private ReOrgOp _op;
	
	private ReorgOp() {
		//default constructor for clone
	}
	
	public ReorgOp(String l, DataType dt, ValueType vt, ReOrgOp o, Hop inp) 
	{
		super(l, dt, vt);
		_op = o;
		getInput().add(0, inp);
		inp.getParent().add(this);

		//compute unknown dims and nnz
		refreshSizeInformation();
	}
	
	public ReorgOp(String l, DataType dt, ValueType vt, ReOrgOp o, List<Hop> inp) 
	{
		super(l, dt, vt);
		_op = o;
		
		for( int i=0; i<inp.size(); i++ ) {
			Hop in = inp.get(i);
			getInput().add(i, in);
			in.getParent().add(this);
		}

		//compute unknown dims and nnz
		refreshSizeInformation();
	}

	public ReOrgOp getOp() {
		return _op;
	}
	
	@Override
	public String getOpString() {
		return "r(" + _op.toString() + ")";
	}
	
	@Override
	public boolean isGPUEnabled() {
		if(!DMLScript.USE_ACCELERATOR)
			return false;
		switch( _op ) {
			case TRANS: {
				if( getDim1()==1 && getDim2()==1 )
					return false; //if input of size 1x1, avoid unnecessary transpose
				//if input is already a transpose, avoid redundant transpose ops
				return !(getInput().get(0) instanceof ReorgOp)
					|| ((ReorgOp) getInput().get(0)).getOp() != ReOrgOp.TRANS;
			}
			case RESHAPE: {
				return true;
			}
			case DIAG:
			case REV:
			case ROLL:
			case SORT:
				return false;
			default:
				throw new RuntimeException("Unsupported operator:" + _op.name());
		}
	}
	
	@Override
	public boolean isMultiThreadedOpType() {
		return _op == ReOrgOp.TRANS
			|| _op == ReOrgOp.SORT
			|| _op == ReOrgOp.REV;
	}

	@Override
	public Lop constructLops()
	{
		//return already created lops
		if( getLops() != null )
			return getLops();

		ExecType et = optFindExecType();
		
		switch( _op )
		{
			case TRANS:
			{
				Lop lin = getInput().get(0).constructLops();
				if( lin instanceof Transform && ((Transform)lin).getOp()==ReOrgOp.TRANS )
					setLops(lin.getInputs().get(0)); //if input is already a transpose, avoid redundant transpose ops
				else if( getDim1()==1 && getDim2()==1 )
					setLops(lin); //if input of size 1x1, avoid unnecessary transpose
				else { //general case
					int k = OptimizerUtils.getConstrainedNumThreads(_maxNumThreads);
					Transform transform1 = new Transform(lin, _op, getDataType(), getValueType(), et, k);
					setOutputDimensions(transform1);
					setLineNumbers(transform1);
					setLops(transform1);
				}
				break;
			}
			case DIAG: {
				Transform transform1 = new Transform(
						getInput().get(0).constructLops(),
						_op, getDataType(), getValueType(), et);
				setOutputDimensions(transform1);
				setLineNumbers(transform1);
				setLops(transform1);
				break;
				}
			case REV: {
				long numel = getDim1() * getDim2();
				int k = (numel < 3000_000) ?
						1 : OptimizerUtils.getConstrainedNumThreads(_maxNumThreads);
				Transform transform1 = new Transform(
					getInput().get(0).constructLops(),
					_op, getDataType(), getValueType(), et, k);
				setOutputDimensions(transform1);
				setLineNumbers(transform1);
				setLops(transform1);
				break;
			}
			case ROLL: {
				Lop[] linputs = new Lop[2]; //input, shift
				for (int i = 0; i < 2; i++)
					linputs[i] = getInput().get(i).constructLops();

				Transform transform1 = new Transform(linputs, _op, getDataType(), getValueType(), et, 1);

				setOutputDimensions(transform1);
				setLineNumbers(transform1);
				setLops(transform1);
				break;
			}
			case RESHAPE: {
				Lop[] linputs = new Lop[5]; //main, rows, cols, dims, byrow
				for (int i = 0; i < 5; i++)
					linputs[i] = getInput().get(i).constructLops();
				_outputEmptyBlocks = (et == ExecType.SPARK &&
						!OptimizerUtils.allowsToFilterEmptyBlockOutputs(this));
				Transform transform1 = new Transform(linputs,
					_op, getDataType(), getValueType(), _outputEmptyBlocks, et);
				setOutputDimensions(transform1);
				setLineNumbers(transform1);

				setLops(transform1);
				break;
			}
			case SORT: {
				Lop[] linputs = new Lop[4]; //input, by, desc, ixret
				for (int i = 0; i < 4; i++)
					linputs[i] = getInput().get(i).constructLops();
				Hop by = getInput().get(2);
				Transform transform1;

				if( et==ExecType.SPARK ) {
					boolean sortRewrite = !FORCE_DIST_SORT_INDEXES 
						&& isSortSPRewriteApplicable() && by.getDataType().isScalar();
					transform1 = new Transform(linputs, ReOrgOp.SORT,
						getDataType(), getValueType(), et, sortRewrite, 1); //#threads = 1
				}
				else { //CP
					int k = OptimizerUtils.getConstrainedNumThreads(_maxNumThreads);
					transform1 = new Transform( linputs, ReOrgOp.SORT, 
						getDataType(), getValueType(), et, false, k);
				}

				setOutputDimensions(transform1);
				setLineNumbers(transform1);
				setLops(transform1);
				break;
			}
			
			default:
				throw new HopsException("Unsupported lops construction for operation type '"+_op+"'.");
		}
		
		//add reblock/checkpoint lops if necessary
		constructAndSetLopsDataFlowProperties();

		return getLops();
	}


	@Override
	public void computeMemEstimate(MemoTable memo){
		if(_op == ReOrgOp.TRANS && getInput().get(0).isCompressedOutput() )
			_outputMemEstimate = getInput().get(0).getCompressedSize();
		else
			super.computeMemEstimate(memo);
	}

	@Override
	protected double computeOutputMemEstimate( long dim1, long dim2, long nnz ) {
		//no dedicated mem estimation per op type, because always propagated via refreshSizeInformation
		double sparsity = OptimizerUtils.getSparsity(dim1, dim2, nnz);
		return OptimizerUtils.estimateSizeExactSparsity(dim1, dim2, sparsity, getDataType());
	}
	
	@Override
	protected double computeIntermediateMemEstimate( long dim1, long dim2, long nnz )
	{
		if( _op == ReOrgOp.SORT ) {
			Hop ixreturn = getInput().get(3);	
			if( !(ixreturn instanceof LiteralOp && !HopRewriteUtils.getBooleanValueSafe((LiteralOp)ixreturn)
				 && (dim2==1 || nnz==0) ) ) //NOT early abort case 
			{
				//Version 2: memory requirements for temporary index int[] array,
				//(temporary double[] array already covered by output)
				return dim1 * 4;
				
				//Version 1: memory requirements for temporary index Integer[] array
				//8-16 (12) bytes for object, 4byte int payload, 4-8 (8) byte pointers.
				//return dim1 * 24; 
			}
		}
		
		//default: no intermediate memory requirements
		return 0;
	}
	
	@Override
	protected DataCharacteristics inferOutputCharacteristics( MemoTable memo )
	{
		DataCharacteristics ret = null;
	
		Hop input = getInput().get(0);
		DataCharacteristics dc = memo.getAllInputStats(input);
	
		switch(_op) 
		{
			case TRANS:{
				// input is a [k1,k2] matrix and output is a [k2,k1] matrix
				// #nnz in output is exactly the same as in input
				if( dc.dimsKnown() )
					ret = new MatrixCharacteristics(dc.getCols(), dc.getRows(), -1, dc.getNonZeros());
				break;
			}
			case REV:
			case ROLL: {
				// dims and nnz are exactly the same as in input
				if (dc.dimsKnown())
					ret = new MatrixCharacteristics(dc.getRows(), dc.getCols(), -1, dc.getNonZeros());
				break;
			}
			case DIAG: {
				// NOTE: diag is overloaded according to the number of columns of the input
				
				long k = dc.getRows();
				
				// CASE a) DIAG V2M
				// input is a [1,k] or [k,1] matrix, and output is [k,k] matrix
				// #nnz in output is in the worst case k => sparsity = 1/k
				if( k == 1 )
					ret = new MatrixCharacteristics(k, k, -1, ((dc.getNonZeros()>=0) ? dc.getNonZeros() : k));
				
				// CASE b) DIAG M2V
				// input is [k,k] matrix and output is [k,1] matrix
				// #nnz in the output is likely to be k (a dense matrix)
				if( k > 1 )
					ret = new MatrixCharacteristics(k, 1, -1, ((dc.getNonZeros()>=0) ? Math.min(k,dc.getNonZeros()) : k));
				
				break;
			}
			case RESHAPE: {
				// input is a [k1,k2] matrix and output is a [k3,k4] matrix with k1*k2=k3*k4, except for
				// special cases where an input or output dimension is zero (i.e., 0x5 -> 1x0 is valid)
				// #nnz in output is exactly the same as in input
				if( dc.dimsKnown() ) {
					if( rowsKnown() && getDim1()!=0 )
						ret = new MatrixCharacteristics(getDim1(), dc.getRows()*dc.getCols()/getDim1(), -1, dc.getNonZeros());
					else if( colsKnown() && getDim2()!=0 ) 
						ret = new MatrixCharacteristics(dc.getRows()*dc.getCols()/getDim2(), getDim2(), -1, dc.getNonZeros());
					else if( dimsKnown() )
						ret = new MatrixCharacteristics(getDim1(), getDim2(), -1, -1);
				}
				break;
			}
			case SORT: {
				// input is a [k1,k2] matrix and output is a [k1,k3] matrix, where k3=k2 if no index return;
				// otherwise k3=1 (for the index vector)
				Hop input4 = getInput().get(3); //indexreturn
				boolean unknownIxRet = !(input4 instanceof LiteralOp);
				
				if( !unknownIxRet ) {
					boolean ixret = HopRewriteUtils.getBooleanValueSafe((LiteralOp)input4);
					long dim2 = ixret ? 1 : dc.getCols();
					long nnz = ixret ? dc.getRows() : dc.getNonZeros();
					ret = new MatrixCharacteristics(dc.getRows(), dim2, -1, nnz);
				}
				else {
					ret = new MatrixCharacteristics(dc.getRows(), -1, -1, -1);
				}
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
	protected ExecType optFindExecType(boolean transitive) {
		
		checkAndSetForcedPlatform();
		
		if( _etypeForced != null )
		{
			_etype = _etypeForced;
		}
		else 
		{	
			if ( OptimizerUtils.isMemoryBasedOptLevel() ) {
				_etype = findExecTypeByMemEstimate();
			}
			// Choose CP, if the input dimensions are below threshold or if the input is a vector
			else if ( getInput().get(0).areDimsBelowThreshold() || getInput().get(0).isVector() )
			{
				_etype = ExecType.CP;
			}
			else 
			{
				_etype = ExecType.SPARK;
			}
			
			//check for valid CP dimensions and matrix size
			checkAndSetInvalidCPDimsAndSize();
		}

		//mark for recompile (forever)
		setRequiresRecompileIfNecessary();
		
		return _etype;
	}
	
	@Override
	public void refreshSizeInformation()
	{
		Hop input1 = getInput().get(0);
		
		switch(_op) 
		{
			case TRANS:
			{
				// input is a [k1,k2] matrix and output is a [k2,k1] matrix
				// #nnz in output is exactly the same as in input
				setDim1(input1.getDim2());
				setDim2(input1.getDim1());
				setNnz(input1.getNnz());
				break;
			}
			case REV:
			case ROLL:
			{
				// dims and nnz are exactly the same as in input
				setDim1(input1.getDim1());
				setDim2(input1.getDim2());
				setNnz(input1.getNnz());
				break;
			}
			case DIAG:
			{
				// NOTE: diag is overloaded according to the number of columns of the input
				
				long k = input1.getDim1(); 
				setDim1(k);
				
				// CASE a) DIAG_V2M
				// input is a [1,k] or [k,1] matrix, and output is [k,k] matrix
				// #nnz in output is in the worst case k => sparsity = 1/k
				if( input1.getDim2()==1 ) {
					setDim2(k);
					setNnz( (input1.getNnz()>=0) ? input1.getNnz() : k );
				}
				
				// CASE b) DIAG_M2V
				// input is [k,k] matrix and output is [k,1] matrix
				// #nnz in the output is likely to be k (a dense matrix)
				if( input1.getDim2()>1 ){
					setDim2(1);
					setNnz( (input1.getNnz()>=0) ? Math.min(k,input1.getNnz()) : k );
				}
				
				break;
			}
			case RESHAPE:
			{
				if (_dataType != DataType.TENSOR) {
					// input is a [k1,k2] matrix and output is a [k3,k4] matrix with k1*k2=k3*k4
					// #nnz in output is exactly the same as in input
					Hop input2 = getInput().get(1); //rows
					Hop input3 = getInput().get(2); //cols
					refreshRowsParameterInformation(input2); //refresh rows
					refreshColsParameterInformation(input3); //refresh cols
					setNnz(input1.getNnz());
					if (!dimsKnown() && input1.dimsKnown()) { //reshape allows to infer dims, if input and 1 dim known
						if (rowsKnown() && getDim1()!=0)
							setDim2(input1.getLength() / getDim1());
						else if (colsKnown() && getDim2()!=0)
							setDim1(input1.getLength() / getDim2());
					}
				} else {
					// TODO size information for tensor
					setNnz(input1.getNnz());
				}

				break;
			}
			case SORT:
			{
				// input is a [k1,k2] matrix and output is a [k1,k3] matrix, where k3=k2 if no index return;
				// otherwise k3=1 (for the index vector)
				Hop input4 = getInput().get(3); //indexreturn
				boolean unknownIxRet = !(input4 instanceof LiteralOp);
				
				setDim1(input1.getDim1());
				if( !unknownIxRet ) {
					boolean ixret = HopRewriteUtils.getBooleanValueSafe((LiteralOp)input4);
					setDim2(ixret ? 1 : input1.getDim2());
					setNnz(ixret ? input1.getDim1() : input1.getNnz());
				}
				else {
					setDim2(-1);
					setNnz(-1);
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
		ret._op = _op;
		ret._maxNumThreads = _maxNumThreads;
		
		return ret;
	}
	
	@Override
	public boolean compare( Hop that ) {
		if( !(that instanceof ReorgOp) )
			return false;
		
		ReorgOp that2 = (ReorgOp)that;
		boolean ret =  (_op == that2._op)
			&& (_maxNumThreads == that2._maxNumThreads)
			&& (getInput().size()==that.getInput().size());
		
		//compare all childs (see reshape, sort)
		if( ret ) //sizes matched
			for( int i=0; i<_input.size(); i++ )
				ret &= getInput().get(i) == that2.getInput().get(i);
		
		return ret;
	}

	/**
	 * This will check if there is sufficient memory locally (twice the size of second matrix, for original and sort data), and remotely (size of second matrix (sorted data)).  
	 * @return true if sufficient memory locally
	 */
	private boolean isSortSPRewriteApplicable() 
	{
		boolean ret = false;
		Hop input = getInput().get(0);
		
		//note: both cases (partitioned matrix, and sorted double array), require to
		//fit the broadcast twice into the local memory budget. Also, the memory 
		//constraint only needs to take the rhs into account because the output is 
		//guaranteed to be an aggregate of <=16KB
		
		double size = input.dimsKnown() ? 
				OptimizerUtils.estimateSize(input.getDim1(), 1) : //dims known and estimate fits
					input.getOutputMemEstimate();                 //dims unknown but worst-case estimate fits
		
		if( OptimizerUtils.checkSparkBroadcastMemoryBudget(size) ) {
			ret = true;
		}
		
		return ret;
	}
}
