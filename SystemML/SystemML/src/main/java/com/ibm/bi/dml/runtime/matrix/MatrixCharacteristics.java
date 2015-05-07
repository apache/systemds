/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix;

import java.util.HashMap;

import com.ibm.bi.dml.lops.MMTSJ.MMTSJType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.instructions.mr.AggregateBinaryInstruction;
import com.ibm.bi.dml.runtime.instructions.mr.AggregateInstruction;
import com.ibm.bi.dml.runtime.instructions.mr.AggregateUnaryInstruction;
import com.ibm.bi.dml.runtime.instructions.mr.AppendInstruction;
import com.ibm.bi.dml.runtime.instructions.mr.BinaryInstruction;
import com.ibm.bi.dml.runtime.instructions.mr.BinaryMInstruction;
import com.ibm.bi.dml.runtime.instructions.mr.BinaryMRInstructionBase;
import com.ibm.bi.dml.runtime.instructions.mr.CM_N_COVInstruction;
import com.ibm.bi.dml.runtime.instructions.mr.CombineBinaryInstruction;
import com.ibm.bi.dml.runtime.instructions.mr.CombineTernaryInstruction;
import com.ibm.bi.dml.runtime.instructions.mr.CombineUnaryInstruction;
import com.ibm.bi.dml.runtime.instructions.mr.CumsumAggregateInstruction;
import com.ibm.bi.dml.runtime.instructions.mr.DataGenMRInstruction;
import com.ibm.bi.dml.runtime.instructions.mr.GroupedAggregateInstruction;
import com.ibm.bi.dml.runtime.instructions.mr.MMTSJMRInstruction;
import com.ibm.bi.dml.runtime.instructions.mr.MRInstruction;
import com.ibm.bi.dml.runtime.instructions.mr.MapMultChainInstruction;
import com.ibm.bi.dml.runtime.instructions.mr.MatrixReshapeMRInstruction;
import com.ibm.bi.dml.runtime.instructions.mr.PMMJMRInstruction;
import com.ibm.bi.dml.runtime.instructions.mr.QuaternaryInstruction;
import com.ibm.bi.dml.runtime.instructions.mr.RandInstruction;
import com.ibm.bi.dml.runtime.instructions.mr.RangeBasedReIndexInstruction;
import com.ibm.bi.dml.runtime.instructions.mr.RangeBasedReIndexInstruction.IndexRange;
import com.ibm.bi.dml.runtime.instructions.mr.ReblockInstruction;
import com.ibm.bi.dml.runtime.instructions.mr.RemoveEmptyMRInstruction;
import com.ibm.bi.dml.runtime.instructions.mr.ReorgInstruction;
import com.ibm.bi.dml.runtime.instructions.mr.ReplicateInstruction;
import com.ibm.bi.dml.runtime.instructions.mr.ScalarInstruction;
import com.ibm.bi.dml.runtime.instructions.mr.SeqInstruction;
import com.ibm.bi.dml.runtime.instructions.mr.TernaryInstruction;
import com.ibm.bi.dml.runtime.instructions.mr.UnaryInstruction;
import com.ibm.bi.dml.runtime.instructions.mr.UnaryMRInstructionBase;
import com.ibm.bi.dml.runtime.instructions.mr.ZeroOutInstruction;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateBinaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateUnaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.ReorgOperator;


public class MatrixCharacteristics
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
	                                         "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
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
	
	public boolean rowsKnown() {
		return ( numRows > 0 );
	}

	public boolean colsKnown() {
		return ( numColumns > 0 );
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
			dimOut.set(in_dim1.numRows, in_dim1.numColumns+in_dim2.numColumns, in_dim1.numRowsPerBlock, in_dim2.numColumnsPerBlock);
		}
		else if(ins instanceof CumsumAggregateInstruction)
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
			//output size independent of chain type
			QuaternaryInstruction realIns=(QuaternaryInstruction)ins;
			MatrixCharacteristics mc1 = dims.get(realIns.getInput1());
			dimOut.set(1, 1, mc1.numRowsPerBlock, mc1.numColumnsPerBlock);	
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
				|| ins instanceof SeqInstruction
				) {
			DataGenMRInstruction dataIns=(DataGenMRInstruction)ins;
			dimOut.set(dims.get(dataIns.getInput()));
		}
		else if(ins instanceof ScalarInstruction 
				|| ins instanceof AggregateInstruction
				||(ins instanceof UnaryInstruction && !(ins instanceof MMTSJMRInstruction))
				|| ins instanceof ReplicateInstruction
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
			dimOut.set( (tstype==MMTSJType.LEFT)? mc.numColumns : mc.numRows,
					     (tstype==MMTSJType.LEFT)? mc.numColumns : mc.numRows,
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
		else if(ins instanceof BinaryInstruction || ins instanceof BinaryMInstruction || ins instanceof CombineBinaryInstruction )
		{
			BinaryMRInstructionBase realIns=(BinaryMRInstructionBase)ins;
			dimOut.set(dims.get(realIns.input1));
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
			RangeBasedReIndexInstruction realIns=(RangeBasedReIndexInstruction)ins;
			MatrixCharacteristics in_dim=dims.get(realIns.input);
			IndexRange ixrange = realIns.getIndexRange(); 
			long nrow=ixrange.rowEnd-ixrange.rowStart+1;
			long ncol=ixrange.colEnd-ixrange.colStart+1;
			dimOut.set(nrow, ncol, in_dim.numRowsPerBlock, in_dim.numColumnsPerBlock);
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
