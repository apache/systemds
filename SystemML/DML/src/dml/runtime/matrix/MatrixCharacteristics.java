package dml.runtime.matrix;

import java.util.HashMap;

import dml.runtime.instructions.MRInstructions.AggregateBinaryInstruction;
import dml.runtime.instructions.MRInstructions.AggregateInstruction;
import dml.runtime.instructions.MRInstructions.AggregateUnaryInstruction;
import dml.runtime.instructions.MRInstructions.AppendInstruction;
import dml.runtime.instructions.MRInstructions.BinaryInstruction;
import dml.runtime.instructions.MRInstructions.BinaryMRInstructionBase;
import dml.runtime.instructions.MRInstructions.CM_N_COVInstruction;
import dml.runtime.instructions.MRInstructions.CombineBinaryInstruction;
import dml.runtime.instructions.MRInstructions.CombineTertiaryInstruction;
import dml.runtime.instructions.MRInstructions.GroupedAggregateInstruction;
import dml.runtime.instructions.MRInstructions.MRInstruction;
import dml.runtime.instructions.MRInstructions.RangeBasedReIndexInstruction;
import dml.runtime.instructions.MRInstructions.TertiaryInstruction;
import dml.runtime.instructions.MRInstructions.UnaryMRInstructionBase;
import dml.runtime.instructions.MRInstructions.RandInstruction;
import dml.runtime.instructions.MRInstructions.ReblockInstruction;
import dml.runtime.instructions.MRInstructions.ReorgInstruction;
import dml.runtime.instructions.MRInstructions.ScalarInstruction;
import dml.runtime.instructions.MRInstructions.UnaryInstruction;
import dml.runtime.matrix.operators.AggregateBinaryOperator;
import dml.runtime.matrix.operators.AggregateUnaryOperator;
import dml.runtime.matrix.operators.ReorgOperator;
import dml.utils.DMLRuntimeException;
import dml.utils.DMLUnsupportedOperationException;

public class MatrixCharacteristics{
	public long numRows=-1;
	public long numColumns=-1;
	public int numRowsPerBlock=1;
	public int numColumnsPerBlock=1;
	public long nonZeros=-1;

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
		nonZeros = nnz;
	}
	
	public void set(MatrixCharacteristics that) {
		this.numRows=that.numRows;
		this.numColumns=that.numColumns;
		this.numRowsPerBlock=that.numRowsPerBlock;
		this.numColumnsPerBlock=that.numColumnsPerBlock;
		this.nonZeros = that.nonZeros;
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
		return "["+numRows+" x "+numColumns+", nnz="+nonZeros
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
		nonZeros = nnz;
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
				|| ins instanceof RandInstruction)
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
			long ncol=realIns.indexRange.colEnd-realIns.indexRange.colEnd+1;
			dim_out.set(nrow, ncol, in_dim.numRowsPerBlock, in_dim.numColumnsPerBlock);
		}
		else { 
			/*
			 * if ins is none of the above cases then we assume that dim_out dimensions are unknown
			 */
			dim_out.numRows = -1;
			dim_out.numColumns = -1;
			dim_out.numRowsPerBlock=1;
			dim_out.numColumnsPerBlock=1;
		}
	}

}
