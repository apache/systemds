package dml.runtime.instructions.MRInstructions;

import dml.runtime.instructions.Instruction;
import dml.runtime.instructions.InstructionUtils;
import dml.runtime.matrix.io.MatrixBlock;
import dml.runtime.matrix.io.MatrixBlock1D;
import dml.runtime.matrix.io.MatrixIndexes;
import dml.runtime.matrix.io.MatrixValue;
import dml.runtime.matrix.io.OperationsOnMatrixValues;
import dml.runtime.matrix.mapred.CachedValueMap;
import dml.runtime.matrix.mapred.IndexedMatrixValue;
import dml.runtime.matrix.operators.Operator;
import dml.utils.DMLRuntimeException;
import dml.utils.DMLUnsupportedOperationException;

public class CombineTertiaryInstruction extends TertiaryInstruction{

	public CombineTertiaryInstruction(Operator op, byte in1, byte in2,
			byte in3, byte out, String istr) {
		super(op, in1, in2, in3, out, istr);
		mrtype = MRINSTRUCTION_TYPE.CombineTertiary;
	}

	public static Instruction parseInstruction ( String str ) throws DMLRuntimeException {
		
		// example instruction string - ctabletransform:::0:DOUBLE:::1:DOUBLE:::2:DOUBLE:::3:DOUBLE 
		InstructionUtils.checkNumFields ( str, 4 );
		
		String[] parts = InstructionUtils.getInstructionParts ( str );
		
		byte in1, in2, in3, out;
		String opcode = parts[0];
		in1 = Byte.parseByte(parts[1]);
		in2 = Byte.parseByte(parts[2]);
		in3 = Byte.parseByte(parts[3]);
		out = Byte.parseByte(parts[4]);
		
		if ( opcode.equalsIgnoreCase("combinetertiary") ) {
			return new CombineTertiaryInstruction(null, in1, in2, in3, out, str);
		} 
		return null;
	}
	
	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass,
			CachedValueMap cachedValues, IndexedMatrixValue tempValue, IndexedMatrixValue zeroInput, 
			int blockRowFactor, int blockColFactor)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		throw new DMLRuntimeException("CombineTertiaryInstruction.processInstruction should never be called!");
	}
}
