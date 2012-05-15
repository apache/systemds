package com.ibm.bi.dml.runtime.instructions.MRInstructions;

import com.ibm.bi.dml.runtime.functionobjects.MaxIndex;
import com.ibm.bi.dml.runtime.functionobjects.SwapIndex;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.io.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.io.OperationsOnMatrixValues;
import com.ibm.bi.dml.runtime.matrix.mapred.CachedValueMap;
import com.ibm.bi.dml.runtime.matrix.mapred.IndexedMatrixValue;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.matrix.operators.ReorgOperator;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.DMLUnsupportedOperationException;


public class ReorgInstruction extends UnaryMRInstructionBase {

	public ReorgInstruction(Operator op, byte in, byte out, String istr)
	{
		super(op, in, out);
		mrtype = MRINSTRUCTION_TYPE.Reorg;
		instString = istr;
	}
	
	public static Instruction parseInstruction ( String str ) throws DMLRuntimeException {
		
		InstructionUtils.checkNumFields ( str, 2 );
		
		String[] parts = InstructionUtils.getInstructionParts ( str );
		
		byte in, out;
		String opcode = parts[0];
		in = Byte.parseByte(parts[1]);
		out = Byte.parseByte(parts[2]);
		
		if ( opcode.equalsIgnoreCase("r'") ) {
			return new ReorgInstruction(new ReorgOperator(SwapIndex.getSwapIndexFnObject()), in, out, str);
		} 
		
		else if ( opcode.equalsIgnoreCase("rdiagV2M") ) {
			return new ReorgInstruction(new ReorgOperator(MaxIndex.getMaxIndexFnObject()), in, out, str);
		} 
		
		else {
			throw new DMLRuntimeException("Unknown opcode while parsing a ReorgInstruction: " + str);
		}
		
	}

	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass,
			CachedValueMap cachedValues, IndexedMatrixValue tempValue, IndexedMatrixValue zeroInput, 
			int blockRowFactor, int blockColFactor)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		
		IndexedMatrixValue in=cachedValues.getFirst(input);
		if(in==null)
			return;
		/*
		MatrixCharacteristics dim=dimensions.get(input);
		
		int startRow=0, startColumn=0, length=0;
		
		
		if(((ReorgOperator)optr).fn==Reorg.SupportedOperation.REORG_MATRIX_DIAG)
		{
			double rowMin=in.getIndexes().getRowIndex()*dim.numRowsPerBlock;
			double columnMin=in.getIndexes().getColumnIndex()*dim.numColumnsPerBlock;
			double left=Math.max(rowMin, columnMin);
			double right=Math.min(rowMin+in.getValue().getNumRows(), columnMin+in.getValue().getNumColumns());
			if(left>=right)
				return;	
			startRow=(int)(left-rowMin);
			startColumn=(int)(left-columnMin);
			length=(int)(right-left);
		}*/
		
		//TODO: cannot handle diag matrix to vector
		int startRow=0, startColumn=0, length=0;
		
		//allocate space for the output value
		IndexedMatrixValue out;
		if(input==output)
			out=tempValue;
		else
			out=cachedValues.holdPlace(output, valueClass);
		
		//process instruction
		OperationsOnMatrixValues.performReorg(in.getIndexes(), in.getValue(), 
				out.getIndexes(), out.getValue(), ((ReorgOperator)optr),
				startRow, startColumn, length);
		
		//put the output value in the cache
		if(out==tempValue)
			cachedValues.add(output, out);
		
	}


}
