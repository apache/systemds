package dml.runtime.instructions.MRInstructions;

import dml.runtime.instructions.Instruction;
import dml.runtime.instructions.InstructionUtils;
import dml.runtime.matrix.io.MatrixValue;
import dml.runtime.matrix.io.OperationsOnMatrixValues;
import dml.runtime.matrix.mapred.CachedValueMap;
import dml.runtime.matrix.mapred.IndexedMatrixValue;
import dml.runtime.matrix.operators.Operator;
import dml.runtime.matrix.operators.UnaryOperator;
import dml.utils.DMLRuntimeException;
import dml.utils.DMLUnsupportedOperationException;

public class UnaryInstruction extends UnaryMRInstructionBase {

	public UnaryInstruction(Operator op, byte in, byte out, String istr)
	{
		super(op, in, out);
		mrtype = MRINSTRUCTION_TYPE.Unary;
		instString = istr;
	}
	
	public static Instruction parseInstruction ( String str ) throws DMLRuntimeException {
		
		String opcode = InstructionUtils.getOpCode(str);
	 
		InstructionUtils.checkNumFields ( str, 2 );
		
		String[] parts = InstructionUtils.getInstructionParts ( str );
		byte in, out;
		in = Byte.parseByte(parts[1]);
		out = Byte.parseByte(parts[2]);
		
		// Currently, UnaryInstructions are used primarily for executing Builtins like SIN(A), LOG(A,2)
		if ( InstructionUtils.isBuiltinFunction(opcode) ) {
			UnaryOperator unary = new UnaryOperator(dml.runtime.functionobjects.Builtin.getBuiltinFnObject(opcode));
			return new UnaryInstruction(unary, in, out, str);
		}
		return null;
	}

	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass,
			CachedValueMap cachedValues, IndexedMatrixValue tempValue,
			IndexedMatrixValue zeroInput, int blockRowFactor, int blockColFactor)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		
		IndexedMatrixValue in=cachedValues.get(input);
		if(in==null)
			return;
		
		//allocate space for the output value
		IndexedMatrixValue out;
		if(input==output)
			out=tempValue;
		else
			out=cachedValues.holdPlace(output, valueClass);
		
		//process instruction
		out.getIndexes().setIndexes(in.getIndexes());
		OperationsOnMatrixValues.performUnaryIgnoreIndexes(in.getValue(), out.getValue(), (UnaryOperator)optr);
		
		//put the output value in the cache
		if(out==tempValue)
			cachedValues.set(output, out);
		
	}

}
