package com.ibm.bi.dml.runtime.instructions.MRInstructions;

import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.RangeBasedReIndexInstruction.IndexRange;
import com.ibm.bi.dml.runtime.matrix.io.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.io.OperationsOnMatrixValues;
import com.ibm.bi.dml.runtime.matrix.mapred.CachedValueMap;
import com.ibm.bi.dml.runtime.matrix.mapred.IndexedMatrixValue;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.matrix.operators.ReIndexOperator;
import com.ibm.bi.dml.runtime.util.UtilFunctions;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.DMLUnsupportedOperationException;


public class MaskInstruction extends UnaryMRInstructionBase{
	
	public IndexRange indexRange=null;
	private IndexRange tempRange=new IndexRange(-1, -1, -1, -1);
	
	public MaskInstruction(Operator op, byte in, byte out, IndexRange rng, String istr) {
		super(op, in, out);
		mrtype = MRINSTRUCTION_TYPE.Select;
		instString = istr;
		indexRange=rng;
	}
	
	private static long parseStartBoundary(String str)
	{
		if(!str.isEmpty())
			return Long.parseLong(str);
		else
			return 1;
	}
	
	private static long parseEndBoundary(String str)
	{
		if(!str.isEmpty())
			return Long.parseLong(str);
		else
			return Long.MAX_VALUE;
	}
	
	public static Instruction parseInstruction ( String str ) throws DMLRuntimeException {
		
		InstructionUtils.checkNumFields ( str, 3 );
		
		String[] parts = InstructionUtils.getInstructionParts ( str );
		
		byte in, out;
		String opcode = parts[0];
		if(!opcode.equalsIgnoreCase("sel"))
			throw new DMLRuntimeException("Unknown opcode while parsing a Select: " + str);
		in = Byte.parseByte(parts[1]);
		out = Byte.parseByte(parts[2]);
		IndexRange rng=new IndexRange(-1, -1, -1, -1);
		String[] strs=InstructionUtils.getInstructionPartsWithValueType(parts[3]);
		if(strs.length!=4)
			throw new DMLRuntimeException("ill formated range " + parts[3]);
		rng.rowStart=parseStartBoundary(strs[0]);
		rng.rowEnd=parseEndBoundary(strs[1]);
		rng.colStart=parseStartBoundary(strs[2]);
		rng.colEnd=parseEndBoundary(strs[3]);
		return new MaskInstruction(new ReIndexOperator(), in, out, rng, str);
		
	}

	private IndexRange getSelectedRange(IndexedMatrixValue in, int blockRowFactor, int blockColFactor) {
		
		long topBlockRowIndex=UtilFunctions.blockIndexCalculation(indexRange.rowStart, blockRowFactor);
		int topRowInTopBlock=UtilFunctions.cellInBlockCalculation(indexRange.rowStart, blockRowFactor);
		long bottomBlockRowIndex=UtilFunctions.blockIndexCalculation(indexRange.rowEnd, blockRowFactor);
		int bottomRowInBottomBlock=UtilFunctions.cellInBlockCalculation(indexRange.rowEnd, blockRowFactor);
		
		long leftBlockColIndex=UtilFunctions.blockIndexCalculation(indexRange.colStart, blockColFactor);
		int leftColInLeftBlock=UtilFunctions.cellInBlockCalculation(indexRange.colStart, blockColFactor);
		long rightBlockColIndex=UtilFunctions.blockIndexCalculation(indexRange.colEnd, blockColFactor);
		int rightColInRightBlock=UtilFunctions.cellInBlockCalculation(indexRange.colEnd, blockColFactor);
		
		//no overlap
		if(in.getIndexes().getRowIndex()<topBlockRowIndex || in.getIndexes().getRowIndex()>bottomBlockRowIndex
		   || in.getIndexes().getColumnIndex()<leftBlockColIndex || in.getIndexes().getColumnIndex()>rightBlockColIndex)
		{
			tempRange.set(-1,-1,-1,-1);
			return tempRange;
		}
		
		//get the index range inside the block
		tempRange.set(0, in.getValue().getNumRows()-1, 0, in.getValue().getNumColumns()-1);
		if(topBlockRowIndex==in.getIndexes().getRowIndex())
			tempRange.rowStart=topRowInTopBlock;
		if(bottomBlockRowIndex==in.getIndexes().getRowIndex())
			tempRange.rowEnd=bottomRowInBottomBlock;
		if(leftBlockColIndex==in.getIndexes().getColumnIndex())
			tempRange.colStart=leftColInLeftBlock;
		if(rightBlockColIndex==in.getIndexes().getColumnIndex())
			tempRange.colEnd=rightColInRightBlock;
		
		return tempRange;
	}
	
	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass,
			CachedValueMap cachedValues, IndexedMatrixValue tempValue,
			IndexedMatrixValue zeroInput, int blockRowFactor, int blockColFactor)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		
		IndexedMatrixValue in=cachedValues.getFirst(input);
		if(in==null)
			return;
		
		tempRange=getSelectedRange(in, blockRowFactor, blockColFactor);
		if(tempRange.rowStart==-1)
			return;
		
		//allocate space for the output value
		IndexedMatrixValue out;
		if(input==output)
			out=tempValue;
		else
			out=cachedValues.holdPlace(output, valueClass);
		
		//process instruction
		
		OperationsOnMatrixValues.performMask(in.getIndexes(), in.getValue(), 
				out.getIndexes(), out.getValue(), tempRange);
		
		//put the output value in the cache
		if(out==tempValue)
			cachedValues.add(output, out);
		
	}
}
