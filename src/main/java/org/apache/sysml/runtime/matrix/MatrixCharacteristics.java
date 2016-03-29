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


package org.apache.sysml.runtime.matrix;

import java.io.Serializable;
import java.util.HashMap;

import org.apache.sysml.lops.MMTSJ.MMTSJType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.DMLUnsupportedOperationException;
import org.apache.sysml.runtime.instructions.mr.AggregateBinaryInstruction;
import org.apache.sysml.runtime.instructions.mr.AggregateInstruction;
import org.apache.sysml.runtime.instructions.mr.AggregateUnaryInstruction;
import org.apache.sysml.runtime.instructions.mr.AppendInstruction;
import org.apache.sysml.runtime.instructions.mr.BinaryInstruction;
import org.apache.sysml.runtime.instructions.mr.BinaryMInstruction;
import org.apache.sysml.runtime.instructions.mr.BinaryMRInstructionBase;
import org.apache.sysml.runtime.instructions.mr.CM_N_COVInstruction;
import org.apache.sysml.runtime.instructions.mr.CombineBinaryInstruction;
import org.apache.sysml.runtime.instructions.mr.CombineTernaryInstruction;
import org.apache.sysml.runtime.instructions.mr.CombineUnaryInstruction;
import org.apache.sysml.runtime.instructions.mr.CumulativeAggregateInstruction;
import org.apache.sysml.runtime.instructions.mr.DataGenMRInstruction;
import org.apache.sysml.runtime.instructions.mr.GroupedAggregateInstruction;
import org.apache.sysml.runtime.instructions.mr.GroupedAggregateMInstruction;
import org.apache.sysml.runtime.instructions.mr.MMTSJMRInstruction;
import org.apache.sysml.runtime.instructions.mr.MRInstruction;
import org.apache.sysml.runtime.instructions.mr.MapMultChainInstruction;
import org.apache.sysml.runtime.instructions.mr.MatrixReshapeMRInstruction;
import org.apache.sysml.runtime.instructions.mr.PMMJMRInstruction;
import org.apache.sysml.runtime.instructions.mr.ParameterizedBuiltinMRInstruction;
import org.apache.sysml.runtime.instructions.mr.QuaternaryInstruction;
import org.apache.sysml.runtime.instructions.mr.RandInstruction;
import org.apache.sysml.runtime.instructions.mr.RangeBasedReIndexInstruction;
import org.apache.sysml.runtime.instructions.mr.ReblockInstruction;
import org.apache.sysml.runtime.instructions.mr.RemoveEmptyMRInstruction;
import org.apache.sysml.runtime.instructions.mr.ReorgInstruction;
import org.apache.sysml.runtime.instructions.mr.ReplicateInstruction;
import org.apache.sysml.runtime.instructions.mr.ScalarInstruction;
import org.apache.sysml.runtime.instructions.mr.SeqInstruction;
import org.apache.sysml.runtime.instructions.mr.TernaryInstruction;
import org.apache.sysml.runtime.instructions.mr.UaggOuterChainInstruction;
import org.apache.sysml.runtime.instructions.mr.UnaryInstruction;
import org.apache.sysml.runtime.instructions.mr.UnaryMRInstructionBase;
import org.apache.sysml.runtime.instructions.mr.ZeroOutInstruction;
import org.apache.sysml.runtime.matrix.operators.AggregateBinaryOperator;
import org.apache.sysml.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysml.runtime.matrix.operators.ReorgOperator;


public class MatrixCharacteristics implements Serializable
{
	private static final long serialVersionUID = 8300479822915546000L;

	private long numRows = -1;
	private long numColumns = -1;
	private int numRowsPerBlock = 1;
	private int numColumnsPerBlock = 1;
	private long nonZero = -1;
	
	public MatrixCharacteristics() {
	
	}
	
	public MatrixCharacteristics(long nr, long nc, int bnr, int bnc)
	{
		set(nr, nc, bnr, bnc);
	}

	public MatrixCharacteristics(long nr, long nc, int bnr, int bnc, long nnz)
	{
		set(nr, nc, bnr, bnc, nnz);
	}
	
	public MatrixCharacteristics(MatrixCharacteristics that)
	{
		set(that.numRows, that.numColumns, that.numRowsPerBlock, that.numColumnsPerBlock, that.nonZero);
	}

	public void set(long nr, long nc, int bnr, int bnc) {
		numRows = nr;
		numColumns = nc;
		numRowsPerBlock = bnr;
		numColumnsPerBlock = bnc;
	}
	
	public void set(long nr, long nc, int bnr, int bnc, long nnz) {
		numRows = nr;
		numColumns = nc;
		numRowsPerBlock = bnr;
		numColumnsPerBlock = bnc;
		nonZero = nnz;
	}
	
	public void set(MatrixCharacteristics that) {
		numRows = that.numRows;
		numColumns = that.numColumns;
		numRowsPerBlock = that.numRowsPerBlock;
		numColumnsPerBlock = that.numColumnsPerBlock;
		nonZero = that.nonZero;
	}
	
	public long getRows(){
		return numRows;
	}

	public long getCols(){
		return numColumns;
	}
	
	public int getRowsPerBlock() {
		return numRowsPerBlock;
	}
	
	public void setRowsPerBlock( int brlen){
		numRowsPerBlock = brlen;
	} 
	
	public int getColsPerBlock() {
		return numColumnsPerBlock;
	}
	
	public void setColsPerBlock( int bclen){
		numColumnsPerBlock = bclen;
	} 
	
	public long getNumRowBlocks(){
		return (long) Math.ceil((double)getRows() / getRowsPerBlock());
	}
	
	public long getNumColBlocks(){
		return (long) Math.ceil((double)getCols() / getColsPerBlock());
	}
	
	public String toString()
	{
		return "["+numRows+" x "+numColumns+", nnz="+nonZero
		+", blocks ("+numRowsPerBlock+" x "+numColumnsPerBlock+")]";
	}
	
	public void setDimension(long nr, long nc)
	{
		numRows = nr;
		numColumns = nc;
	}
	
	public void setBlockSize(int bnr, int bnc)
	{
		numRowsPerBlock = bnr;
		numColumnsPerBlock = bnc;
	}
	
	public void setNonZeros(long nnz) {
		nonZero = nnz;
	}
	
	public long getNonZeros() {
		return nonZero;
	}
	
	public boolean dimsKnown() {
		return ( numRows > 0 && numColumns > 0 );
	}
	
	public boolean dimsKnown(boolean includeNnz) {
		return ( numRows > 0 && numColumns > 0 && (!includeNnz || nonZero>=0));
	}
	
	public boolean rowsKnown() {
		return ( numRows > 0 );
	}

	public boolean colsKnown() {
		return ( numColumns > 0 );
	}
	
	public boolean nnzKnown() {
		return ( nonZero >= 0 );
	}
	
	public boolean mightHaveEmptyBlocks() 
	{
		long singleBlk =  Math.min(numRows, numRowsPerBlock) 
				        * Math.min(numColumns, numColumnsPerBlock);
		return !nnzKnown() || (nonZero < numRows*numColumns - singleBlk);
	}
	
	public static void reorg(MatrixCharacteristics dim, ReorgOperator op, 
			MatrixCharacteristics dimOut) throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		op.fn.computeDimension(dim, dimOut);
	}
	
	public static void aggregateUnary(MatrixCharacteristics dim, AggregateUnaryOperator op, 
			MatrixCharacteristics dimOut) throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		op.indexFn.computeDimension(dim, dimOut);
	}
	
	public static void aggregateBinary(MatrixCharacteristics dim1, MatrixCharacteristics dim2,
			AggregateBinaryOperator op, MatrixCharacteristics dimOut) 
	throws DMLUnsupportedOperationException
	{
		//set dimension
		dimOut.set(dim1.numRows, dim2.numColumns, dim1.numRowsPerBlock, dim2.numColumnsPerBlock);
	}
	
	public static void computeDimension(HashMap<Byte, MatrixCharacteristics> dims, MRInstruction ins) 
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		MatrixCharacteristics dimOut=dims.get(ins.output);
		if(dimOut==null)
		{
			dimOut=new MatrixCharacteristics();
			dims.put(ins.output, dimOut);
		}
		
		if(ins instanceof ReorgInstruction)
		{
			ReorgInstruction realIns=(ReorgInstruction)ins;
			reorg(dims.get(realIns.input), (ReorgOperator)realIns.getOperator(), dimOut);
		}
		else if(ins instanceof AppendInstruction )
		{
			AppendInstruction realIns = (AppendInstruction)ins;
			MatrixCharacteristics in_dim1 = dims.get(realIns.input1);
			MatrixCharacteristics in_dim2 = dims.get(realIns.input2);
			if( realIns.isCBind() )
				dimOut.set(in_dim1.numRows, in_dim1.numColumns+in_dim2.numColumns, in_dim1.numRowsPerBlock, in_dim2.numColumnsPerBlock);
			else
				dimOut.set(in_dim1.numRows+in_dim2.numRows, in_dim1.numColumns, in_dim1.numRowsPerBlock, in_dim2.numColumnsPerBlock);
		}
		else if(ins instanceof CumulativeAggregateInstruction)
		{
			AggregateUnaryInstruction realIns=(AggregateUnaryInstruction)ins;
			MatrixCharacteristics in = dims.get(realIns.input);
			dimOut.set((long)Math.ceil( (double)in.getRows()/in.getRowsPerBlock()), in.getCols(), in.getRowsPerBlock(), in.getColsPerBlock());
		}
		else if(ins instanceof AggregateUnaryInstruction)
		{
			AggregateUnaryInstruction realIns=(AggregateUnaryInstruction)ins;
			aggregateUnary(dims.get(realIns.input), 
					(AggregateUnaryOperator)realIns.getOperator(), dimOut);
		}
		else if(ins instanceof AggregateBinaryInstruction)
		{
			AggregateBinaryInstruction realIns=(AggregateBinaryInstruction)ins;
			aggregateBinary(dims.get(realIns.input1), dims.get(realIns.input2),
					(AggregateBinaryOperator)realIns.getOperator(), dimOut);
		}
		else if(ins instanceof MapMultChainInstruction)
		{
			//output size independent of chain type
			MapMultChainInstruction realIns=(MapMultChainInstruction)ins;
			MatrixCharacteristics mc1 = dims.get(realIns.getInput1());
			MatrixCharacteristics mc2 = dims.get(realIns.getInput2());
			dimOut.set(mc1.numColumns, mc2.numColumns, mc1.numRowsPerBlock, mc1.numColumnsPerBlock);	
		}
		else if(ins instanceof QuaternaryInstruction)
		{
			QuaternaryInstruction realIns=(QuaternaryInstruction)ins;
			MatrixCharacteristics mc1 = dims.get(realIns.getInput1());
			MatrixCharacteristics mc2 = dims.get(realIns.getInput2());
			MatrixCharacteristics mc3 = dims.get(realIns.getInput3());
			realIns.computeMatrixCharacteristics(mc1, mc2, mc3, dimOut);
		}
		else if(ins instanceof ReblockInstruction)
		{
			ReblockInstruction realIns=(ReblockInstruction)ins;
			MatrixCharacteristics in_dim=dims.get(realIns.input);
			dimOut.set(in_dim.numRows, in_dim.numColumns, realIns.brlen, realIns.bclen, in_dim.nonZero);
		}
		else if( ins instanceof MatrixReshapeMRInstruction )
		{
			MatrixReshapeMRInstruction mrinst = (MatrixReshapeMRInstruction) ins;
			MatrixCharacteristics in_dim=dims.get(mrinst.input);
			dimOut.set(mrinst.getNumRows(),mrinst.getNumColunms(),in_dim.getRowsPerBlock(), in_dim.getColsPerBlock(), in_dim.getNonZeros());
		}
		else if(ins instanceof RandInstruction
				|| ins instanceof SeqInstruction) 
		{
			DataGenMRInstruction dataIns=(DataGenMRInstruction)ins;
			dimOut.set(dims.get(dataIns.getInput()));
		}
		else if( ins instanceof ReplicateInstruction )
		{
			ReplicateInstruction realIns=(ReplicateInstruction)ins;
			realIns.computeOutputDimension(dims.get(realIns.input), dimOut);
		}
		else if( ins instanceof ParameterizedBuiltinMRInstruction ) //before unary
		{
			ParameterizedBuiltinMRInstruction realIns = (ParameterizedBuiltinMRInstruction)ins;
			realIns.computeOutputCharacteristics(dims.get(realIns.input), dimOut);
		}
		else if(ins instanceof ScalarInstruction 
				|| ins instanceof AggregateInstruction
				||(ins instanceof UnaryInstruction && !(ins instanceof MMTSJMRInstruction))
				|| ins instanceof ZeroOutInstruction)
		{
			UnaryMRInstructionBase realIns=(UnaryMRInstructionBase)ins;
			dimOut.set(dims.get(realIns.input));
		}
		else if (ins instanceof MMTSJMRInstruction)
		{
			MMTSJMRInstruction mmtsj = (MMTSJMRInstruction)ins;
			MMTSJType tstype = mmtsj.getMMTSJType();
			MatrixCharacteristics mc = dims.get(mmtsj.input);
			dimOut.set( tstype.isLeft() ? mc.numColumns : mc.numRows,
					    tstype.isLeft() ? mc.numColumns : mc.numRows,
					     mc.numRowsPerBlock, mc.numColumnsPerBlock );
		}
		else if( ins instanceof PMMJMRInstruction )
		{
			PMMJMRInstruction pmmins = (PMMJMRInstruction) ins;
			MatrixCharacteristics mc = dims.get(pmmins.input2);
			dimOut.set( pmmins.getNumRows(),
					     mc.numColumns,
					     mc.numRowsPerBlock, mc.numColumnsPerBlock );
		}
		else if( ins instanceof RemoveEmptyMRInstruction )
		{
			RemoveEmptyMRInstruction realIns=(RemoveEmptyMRInstruction)ins;
			MatrixCharacteristics mc = dims.get(realIns.input1);
			if( realIns.isRemoveRows() )
				dimOut.set(realIns.getOutputLen(), mc.getCols(), mc.numRowsPerBlock, mc.numColumnsPerBlock);
			else
				dimOut.set(mc.getRows(), realIns.getOutputLen(), mc.numRowsPerBlock, mc.numColumnsPerBlock);
		}
		else if(ins instanceof UaggOuterChainInstruction) //needs to be checked before binary
		{
			UaggOuterChainInstruction realIns=(UaggOuterChainInstruction)ins;
			MatrixCharacteristics mc1 = dims.get(realIns.input1);
			MatrixCharacteristics mc2 = dims.get(realIns.input2);
			realIns.computeOutputCharacteristics(mc1, mc2, dimOut);
		}
		else if( ins instanceof GroupedAggregateMInstruction )
		{
			GroupedAggregateMInstruction realIns = (GroupedAggregateMInstruction) ins;
			MatrixCharacteristics mc1 = dims.get(realIns.input1);
			realIns.computeOutputCharacteristics(mc1, dimOut);
		}
		else if(ins instanceof BinaryInstruction || ins instanceof BinaryMInstruction || ins instanceof CombineBinaryInstruction )
		{
			BinaryMRInstructionBase realIns=(BinaryMRInstructionBase)ins;
			MatrixCharacteristics mc1 = dims.get(realIns.input1);
			MatrixCharacteristics mc2 = dims.get(realIns.input2);
			if(    mc1.getRows()>1 && mc1.getCols()==1 
				&& mc2.getRows()==1 && mc2.getCols()>1 ) //outer
			{
				dimOut.set(mc1.getRows(), mc2.getCols(), mc1.getRowsPerBlock(), mc2.getColsPerBlock());
			}
			else { //default case
				dimOut.set(mc1);	
			}
		}
		else if (ins instanceof CombineTernaryInstruction ) {
			TernaryInstruction realIns=(TernaryInstruction)ins;
			dimOut.set(dims.get(realIns.input1));
		}
		else if (ins instanceof CombineUnaryInstruction ) {
			dimOut.set( dims.get(((CombineUnaryInstruction) ins).input));
		}
		else if(ins instanceof CM_N_COVInstruction || ins instanceof GroupedAggregateInstruction )
		{
			dimOut.set(1, 1, 1, 1);
		}
		else if(ins instanceof RangeBasedReIndexInstruction)
		{
			RangeBasedReIndexInstruction realIns = (RangeBasedReIndexInstruction)ins;
			MatrixCharacteristics dimIn = dims.get(realIns.input);
			realIns.computeOutputCharacteristics(dimIn, dimOut);
		}
		else if (ins instanceof TernaryInstruction) {
			TernaryInstruction realIns = (TernaryInstruction)ins;
			MatrixCharacteristics in_dim=dims.get(realIns.input1);
			dimOut.set(realIns.getOutputDim1(), realIns.getOutputDim2(), in_dim.numRowsPerBlock, in_dim.numColumnsPerBlock);
		}
		else { 
			/*
			 * if ins is none of the above cases then we assume that dim_out dimensions are unknown
			 */
			dimOut.numRows = -1;
			dimOut.numColumns = -1;
			dimOut.numRowsPerBlock=1;
			dimOut.numColumnsPerBlock=1;
		}
	}

	@Override
	public boolean equals (Object anObject)
	{
		if (anObject instanceof MatrixCharacteristics)
		{
			MatrixCharacteristics mc = (MatrixCharacteristics) anObject;
			return ((numRows == mc.numRows) && 
					(numColumns == mc.numColumns) && 
					(numRowsPerBlock == mc.numRowsPerBlock) && 
					(numColumnsPerBlock == mc.numColumnsPerBlock) && 
					(nonZero == mc.nonZero)) ;
		}
		else
			return false;
	}
	
	@Override
	public int hashCode()
	{
		return super.hashCode();
	}
}
