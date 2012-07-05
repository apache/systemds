package com.ibm.bi.dml.runtime.matrix;

import java.util.HashMap;

import com.ibm.bi.dml.runtime.instructions.MRInstructions.AggregateBinaryInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.AggregateInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.AggregateUnaryInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.AppendInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.BinaryInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.BinaryMRInstructionBase;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.CM_N_COVInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.CombineBinaryInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.CombineTertiaryInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.GroupedAggregateInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.MRInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.RandInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.RangeBasedReIndexInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.ReblockInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.ReorgInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.ScalarInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.TertiaryInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.UnaryInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.UnaryMRInstructionBase;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.ZeroOutInstruction;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateBinaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateUnaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.ReorgOperator;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.DMLUnsupportedOperationException;


public class MatrixCharacteristics{
	public long numRows=-1;
	public long numColumns=-1;
	public int numRowsPerBlock=1;
	public int numColumnsPerBlock=1;
	public long nonZero=-1;
	
	public MatrixCharacteristics(long nr, long nc, int bnr, int bnc)
	{
		set(nr, nc, bnr, bnc);
	}

	public MatrixCharacteristics(long nr, long nc, int bnr, int bnc, long nnz)
	{
		set(nr, nc, bnr, bnc, nnz);
	}

	public MatrixCharacteristics() {
	}

	public void set(long nr, long nc, int bnr, int bnc) {
		numRows=nr;
		numColumns=nc;
		numRowsPerBlock=bnr;
		numColumnsPerBlock=bnc;
	}
	
	public void set(long nr, long nc, int bnr, int bnc, long nnz) {
		numRows=nr;
		numColumns=nc;
		numRowsPerBlock=bnr;
		numColumnsPerBlock=bnc;
		nonZero = nnz;
	}
	
	public void set(MatrixCharacteristics that) {
		this.numRows=that.numRows;
		this.numColumns=that.numColumns;
		this.numRowsPerBlock=that.numRowsPerBlock;
		this.numColumnsPerBlock=that.numColumnsPerBlock;
		this.nonZero = that.nonZero;
	}
	
	public long get_rows(){
		return numRows;
	}

	public long get_cols(){
		return numColumns;
	}
	
	public int get_rows_per_block() {
		return numRowsPerBlock;
	}
	
	public int get_cols_per_block() {
		return numColumnsPerBlock;
	}
	public String toString()
	{
		return "["+numRows+" x "+numColumns+", nnz="+nonZero
		+", blocks ("+numRowsPerBlock+" x "+numColumnsPerBlock+")]";
	}
	
	public void setDimension(long nr, long nc)
	{
		numRows=nr;
		numColumns=nc;
	}
	
	public void setBlockSize(int bnr, int bnc)
	{
		numRowsPerBlock=bnr;
		numColumnsPerBlock=bnc;
	}
	
	public void setNonZeros(long nnz) {
		nonZero = nnz;
	}
	
	public static void reorg(MatrixCharacteristics dim, ReorgOperator op, 
			MatrixCharacteristics dim_out) throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		op.fn.computeDimension(dim, dim_out);
	}
	
	public static void aggregateUnary(MatrixCharacteristics dim, AggregateUnaryOperator op, 
			MatrixCharacteristics dim_out) throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		op.indexFn.computeDimension(dim, dim_out);
	}
	
	public static void aggregateBinary(MatrixCharacteristics dim1, MatrixCharacteristics dim2,
			AggregateBinaryOperator op, MatrixCharacteristics dim_out) 
	throws DMLUnsupportedOperationException
	{
		//set dimension
		dim_out.set(dim1.numRows, dim2.numColumns, dim1.numRowsPerBlock, dim2.numColumnsPerBlock);
	}
	
	public static void computeDimension(HashMap<Byte, MatrixCharacteristics> dims, MRInstruction ins) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		MatrixCharacteristics dim_out=dims.get(ins.output);
		if(dim_out==null)
		{
			dim_out=new MatrixCharacteristics();
			dims.put(ins.output, dim_out);
		}
		
		if(ins instanceof ReorgInstruction)
		{
			ReorgInstruction realIns=(ReorgInstruction)ins;
			reorg(dims.get(realIns.input), (ReorgOperator)realIns.getOperator(), dim_out);
		}else if(ins instanceof AppendInstruction)
		{
			AppendInstruction realIns = (AppendInstruction)ins;
			MatrixCharacteristics in_dim1 = dims.get(realIns.input1);
			MatrixCharacteristics in_dim2 = dims.get(realIns.input2);
			dim_out.set(in_dim1.numRows, in_dim1.numColumns+in_dim2.numColumns, in_dim1.numRowsPerBlock, in_dim2.numColumnsPerBlock);
		}else if(ins instanceof AggregateUnaryInstruction)
		{
			AggregateUnaryInstruction realIns=(AggregateUnaryInstruction)ins;
			aggregateUnary(dims.get(realIns.input), 
					(AggregateUnaryOperator)realIns.getOperator(), dim_out);
		}else if(ins instanceof AggregateBinaryInstruction)
		{
			AggregateBinaryInstruction realIns=(AggregateBinaryInstruction)ins;
			aggregateBinary(dims.get(realIns.input1), dims.get(realIns.input2),
					(AggregateBinaryOperator)realIns.getOperator(), dim_out);
		}else if(ins instanceof ReblockInstruction)
		{
			ReblockInstruction realIns=(ReblockInstruction)ins;
			MatrixCharacteristics in_dim=dims.get(realIns.input);
			dim_out.set(in_dim.numRows, in_dim.numColumns, realIns.brlen, realIns.bclen);
		}else if(ins instanceof ScalarInstruction 
				|| ins instanceof AggregateInstruction
				|| ins instanceof UnaryInstruction
				|| ins instanceof RandInstruction
				|| ins instanceof ZeroOutInstruction)
		{
			UnaryMRInstructionBase realIns=(UnaryMRInstructionBase)ins;
			dim_out.set(dims.get(realIns.input));
		}
		else if(ins instanceof BinaryInstruction || ins instanceof CombineBinaryInstruction )
		{
			BinaryMRInstructionBase realIns=(BinaryMRInstructionBase)ins;
			dim_out.set(dims.get(realIns.input1));
		}
		else if (ins instanceof CombineTertiaryInstruction ) {
			TertiaryInstruction realIns=(TertiaryInstruction)ins;
			dim_out.set(dims.get(realIns.input1));
		}
		else if(ins instanceof CM_N_COVInstruction || ins instanceof GroupedAggregateInstruction )
		{
			dim_out.set(1, 1, 1, 1);
		}
		else if(ins instanceof RangeBasedReIndexInstruction)
		{
			RangeBasedReIndexInstruction realIns=(RangeBasedReIndexInstruction)ins;
			MatrixCharacteristics in_dim=dims.get(realIns.input);
			long nrow=realIns.indexRange.rowEnd-realIns.indexRange.rowStart+1;
			long ncol=realIns.indexRange.colEnd-realIns.indexRange.colStart+1;
			dim_out.set(nrow, ncol, in_dim.numRowsPerBlock, in_dim.numColumnsPerBlock);
		}else { 
			/*
			 * if ins is none of the above cases then we assume that dim_out dimensions are unknown
			 */
			dim_out.numRows = -1;
			dim_out.numColumns = -1;
			dim_out.numRowsPerBlock=1;
			dim_out.numColumnsPerBlock=1;
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
}
