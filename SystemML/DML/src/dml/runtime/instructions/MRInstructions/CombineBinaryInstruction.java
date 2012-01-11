package dml.runtime.instructions.MRInstructions;

import dml.runtime.instructions.Instruction;
import dml.runtime.instructions.InstructionUtils;
import dml.runtime.matrix.io.MatrixValue;
import dml.runtime.matrix.mapred.CachedValueMap;
import dml.runtime.matrix.mapred.IndexedMatrixValue;
import dml.runtime.matrix.operators.Operator;
import dml.utils.DMLRuntimeException;
import dml.utils.DMLUnsupportedOperationException;

public class CombineBinaryInstruction extends BinaryMRInstructionBase{

	/*
	 * combinebinary:::0:DOUBLE:::1:INT:::2:INT
	 */
	private boolean secondInputIsWeight=true;
	public CombineBinaryInstruction(Operator op, boolean isWeight, byte in1, byte in2, byte out, String istr) {
		super(op, in1, in2, out);
		secondInputIsWeight=isWeight;
		mrtype = MRINSTRUCTION_TYPE.CombineBinary;
		instString = istr;
	}

	public static Instruction parseInstruction ( String str ) throws DMLRuntimeException {
		
		InstructionUtils.checkNumFields ( str, 4 );
		
		String[] parts = InstructionUtils.getInstructionParts ( str );
		
		byte in1, in2, out;
		String opcode = parts[0];
		boolean isWeight=Boolean.parseBoolean(parts[1]);
		in1 = Byte.parseByte(parts[2]);
		in2 = Byte.parseByte(parts[3]);
		out = Byte.parseByte(parts[4]);
		
		if ( opcode.equalsIgnoreCase("combinebinary") ) {
			return new CombineBinaryInstruction(null, isWeight, in1, in2, out, str);
		}else
			return null;
	}
	
	public boolean isSecondInputWeight()
	{
		return secondInputIsWeight;
	}

	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass,
			CachedValueMap cachedValues, IndexedMatrixValue tempValue,
			IndexedMatrixValue zeroInput, int blockRowFactor, int blockColFactor)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		throw new DMLRuntimeException("CombineInstruction.processInstruction should never be called!");
		
	}
	
}
