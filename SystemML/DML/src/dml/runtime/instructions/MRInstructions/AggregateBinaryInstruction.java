package dml.runtime.instructions.MRInstructions;

import dml.runtime.functionobjects.Multiply;
import dml.runtime.functionobjects.Plus;
import dml.runtime.instructions.Instruction;
import dml.runtime.instructions.InstructionUtils;
import dml.runtime.matrix.io.MatrixValue;
import dml.runtime.matrix.io.OperationsOnMatrixValues;
import dml.runtime.matrix.mapred.CachedValueMap;
import dml.runtime.matrix.mapred.IndexedMatrixValue;
import dml.runtime.matrix.operators.AggregateBinaryOperator;
import dml.runtime.matrix.operators.AggregateOperator;
import dml.runtime.matrix.operators.Operator;
import dml.utils.DMLRuntimeException;
import dml.utils.DMLUnsupportedOperationException;

public class AggregateBinaryInstruction extends BinaryMRInstructionBase {

	public AggregateBinaryInstruction(Operator op, byte in1, byte in2, byte out, String istr)
	{
		super(op, in1, in2, out);
		mrtype = MRINSTRUCTION_TYPE.AggregateBinary;
		instString = istr;
	}
	
	public static Instruction parseInstruction ( String str ) throws DMLRuntimeException {
		
		InstructionUtils.checkNumFields ( str, 3 );
		
		String[] parts = InstructionUtils.getInstructionParts ( str );
		
		byte in1, in2, out;
		String opcode = parts[0];
		in1 = Byte.parseByte(parts[1]);
		in2 = Byte.parseByte(parts[2]);
		out = Byte.parseByte(parts[3]);
		
		if ( opcode.equalsIgnoreCase("cpmm") || opcode.equalsIgnoreCase("rmm") ) {
			AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
			AggregateBinaryOperator aggbin = new AggregateBinaryOperator(Multiply.getMultiplyFnObject(), agg);
			return new AggregateBinaryInstruction(aggbin, in1, in2, out, str);
		} 
		else {
			throw new DMLRuntimeException("AggregateBinaryInstruction.parseInstruction():: Unknown opcode " + opcode);
		}
		
		// return null;
	}

	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass,
			CachedValueMap cachedValues, IndexedMatrixValue tempValue,
			IndexedMatrixValue zeroInput, int blockRowFactor, int blockColFactor)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		
		IndexedMatrixValue in1=cachedValues.get(input1);
		IndexedMatrixValue in2=cachedValues.get(input2);
		if(in1==null || in2==null)
			return;
		
		//allocate space for the output value
		IndexedMatrixValue out;
		if(output==input1 || output==input2)
			out=tempValue;
		else
			out=cachedValues.holdPlace(output, valueClass);
		
		//process instruction
		OperationsOnMatrixValues.performAggregateBinary(in1.getIndexes(), in1.getValue(), 
				in2.getIndexes(), in2.getValue(), out.getIndexes(), out.getValue(), 
				((AggregateBinaryOperator)optr));
		
		//put the output value in the cache
		if(out==tempValue)
			cachedValues.set(output, out);
		
	}

}
