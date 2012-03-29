package dml.runtime.instructions.MRInstructions;

import dml.runtime.instructions.Instruction;
import dml.runtime.instructions.InstructionUtils;
import dml.runtime.matrix.io.MatrixValue;
import dml.runtime.matrix.io.OperationsOnMatrixValues;
import dml.runtime.matrix.mapred.CachedValueMap;
import dml.runtime.matrix.mapred.IndexedMatrixValue;
import dml.runtime.matrix.operators.Operator;
import dml.utils.DMLRuntimeException;
import dml.utils.DMLUnsupportedOperationException;
import dml.runtime.functionobjects.OffsetColumnIndex;
import dml.runtime.matrix.operators.ReorgOperator;

public class AppendInstruction extends BinaryMRInstructionBase {
	int offset; 
	public AppendInstruction(Operator op, byte in1, byte in2, int offset, byte out, String istr)
	{
		super(op, in1, in2, out);
		this.offset = offset;
		mrtype = MRINSTRUCTION_TYPE.Append;
		instString = istr;
	}
	
	public static Instruction parseInstruction ( String str ) throws DMLRuntimeException {
		InstructionUtils.checkNumFields ( str, 4 );
		
		String[] parts = InstructionUtils.getInstructionParts ( str );
		
		byte in1, in2, out;
		int offset;
		in1 = Byte.parseByte(parts[1]);
		in2 = Byte.parseByte(parts[2]);
		out = Byte.parseByte(parts[3]);
		offset = (int)(Double.parseDouble(parts[4]));
		
		return new AppendInstruction(new ReorgOperator(OffsetColumnIndex.getOffsetColumnIndexFnObject(-1)), 
									 in1, 
									 in2,
									 offset,
									 out,
									 str);
	}
	
	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass,
			CachedValueMap cachedValues, IndexedMatrixValue tempValue, IndexedMatrixValue zeroInput, 
			int blockRowFactor, int blockColFactor)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		
		boolean isSecondArg = (cachedValues.get(input1) != null) ? false : true;
		
		IndexedMatrixValue in = (isSecondArg) ? cachedValues.get(input2) : cachedValues.get(input1);
			
		if(in==null)
			return;
		
		OffsetColumnIndex off = ((OffsetColumnIndex)((ReorgOperator)optr).fn);
		
		IndexedMatrixValue out;
		if((isSecondArg && input2==output) || (!isSecondArg && input1==output))
			out=tempValue;
		else
			out=cachedValues.holdPlace(output, valueClass);
		
		if(!isSecondArg){
			off.setOffset(0);
			OperationsOnMatrixValues.performAppend(in.getIndexes(), in.getValue(), 
												   out.getIndexes(), out.getValue(), 
												   ((ReorgOperator)optr));
		}else{//if(isSecondArg)
			off.setOffset(offset);
			OperationsOnMatrixValues.performAppend(in.getIndexes(), in.getValue(), 
					   							   out.getIndexes(), out.getValue(), 
					   							   ((ReorgOperator)optr));
		}
	
		//put the output value in the cache
		if(out==tempValue)
			cachedValues.set(output, out);
	}
}
