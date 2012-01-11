package dml.runtime.instructions.MRInstructions;

import dml.runtime.instructions.Instruction;
import dml.runtime.instructions.InstructionUtils;
import dml.runtime.matrix.io.MatrixValue;
import dml.runtime.matrix.mapred.CachedValueMap;
import dml.runtime.matrix.mapred.IndexedMatrixValue;
import dml.runtime.matrix.operators.Operator;
import dml.utils.DMLRuntimeException;
import dml.utils.DMLUnsupportedOperationException;

public class CombineUnaryInstruction extends UnaryMRInstructionBase{

	/*
	 * combineunary:::0:DOUBLE:::1:DOUBLE
	 */
	public CombineUnaryInstruction(Operator op, byte in, byte out, String istr) {
		super(op, in, out);
		mrtype = MRINSTRUCTION_TYPE.CombineUnary;
		instString = istr;
	}

	public static Instruction parseInstruction ( String str ) throws DMLRuntimeException {
		
		InstructionUtils.checkNumFields ( str, 2 );
		
		String[] parts = InstructionUtils.getInstructionParts ( str );
		
		byte in, out;
		String opcode = parts[0];
		in  = Byte.parseByte(parts[1]);
		out = Byte.parseByte(parts[2]);
		
		if ( opcode.equalsIgnoreCase("combineunary") ) {
			return new CombineUnaryInstruction(null, in, out, str);
		}else
			return null;
	}

	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass,
			CachedValueMap cachedValues, IndexedMatrixValue tempValue,
			IndexedMatrixValue zeroInput, int blockRowFactor, int blockColFactor)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		throw new DMLRuntimeException("CombineInstruction.processInstruction should never be called!");
		
	}
	
}
