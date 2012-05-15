package com.ibm.bi.dml.runtime.instructions.MRInstructions;

import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.io.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.mapred.CachedValueMap;
import com.ibm.bi.dml.runtime.matrix.mapred.IndexedMatrixValue;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.DMLUnsupportedOperationException;


public class PickByCountInstruction extends MRInstruction {

	public byte input1; // used for both valuepick and rangepick
	public byte input2; // used only for valuepick
	public double cst; // used only for rangepick
	public boolean isValuePick=true;
	
	/*
	 *  Constructor for valuepick
	 *  valuepick:::0:DOUBLE:::1:DOUBLE:::2:DOUBLE
	 *  0 is data matrix, 1 is the quantile matrix, 2 will have the resulting picked data items 
	 */
	public PickByCountInstruction(Operator op, byte _in1, byte _in2, byte out, String istr) {
		super(op, out);
		input1 = _in1;
		input2 = _in2;
		cst = 0;
		mrtype = MRINSTRUCTION_TYPE.PickByCount;
		instString = istr;
		isValuePick=true;
	}

	/*
	 *  Constructor for rangepick
	 *  rangepick:::0:DOUBLE:::0.25:DOUBLE:::1:DOUBLE
	 *  0 is data matrix, 0.25 is the quantile that needs to be removed from both ends in the PDF, 
	 *  1 will have the resulting picked data items between [Q_1-Q_3]
	 */
	public PickByCountInstruction(Operator op, byte _in1, double _cst, byte out, String istr) {
		super(op, out);
		input1 = _in1;
		cst = _cst;
		mrtype = MRINSTRUCTION_TYPE.PickByCount;
		instString = istr;
		isValuePick=false;
	}

	public static Instruction parseInstruction ( String str ) throws DMLRuntimeException {
		
		InstructionUtils.checkNumFields ( str, 3 );
		
		String opcode = InstructionUtils.getOpCode(str);
		
		if ( opcode.equalsIgnoreCase("valuepick") ) {
			String[] parts = InstructionUtils.getInstructionParts ( str );
			
			byte in1, in2, out;
			in1 = Byte.parseByte(parts[1]);
			in2 = Byte.parseByte(parts[2]);
			out = Byte.parseByte(parts[3]);
			
			return new PickByCountInstruction(null, in1, in2, out, str);
		} 
		else if ( opcode.equalsIgnoreCase("rangepick")) {
			String[] parts = InstructionUtils.getInstructionParts ( str );
			
			byte in1, out;
			double cstant;
			in1 = Byte.parseByte(parts[1]);
			cstant = Double.parseDouble(parts[2]);
			out = Byte.parseByte(parts[3]);
			
			return new PickByCountInstruction(null, in1, cstant, out, str);
		}
		else
			return null;
	}

	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass,
			CachedValueMap cachedValues, IndexedMatrixValue tempValue,
			IndexedMatrixValue zeroInput, int blockRowFactor, int blockColFactor)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		throw new DMLRuntimeException("PickByCountInstruction.processInstruction should never be called!");
		
	}

	@Override
	public byte[] getAllIndexes() throws DMLRuntimeException {
		String opcode = InstructionUtils.getOpCode(instString);		
		if ( opcode.equalsIgnoreCase("valuepick") ) {
			return new byte[]{input1,input2,output};
		}
		else if ( opcode.equalsIgnoreCase("rangepick") ) {
			return new byte[]{input1, output};
		}
		return null;
	}

	@Override
	public byte[] getInputIndexes() throws DMLRuntimeException {
		String opcode = InstructionUtils.getOpCode(instString);		
		if ( opcode.equalsIgnoreCase("valuepick") ) {
			return new byte[]{input1,input2};
		}
		else if ( opcode.equalsIgnoreCase("rangepick") ) {
			return new byte[]{input1};
		}
		return null;
	}
	
/*	boolean isValuePick() {
		String opcode = InstructionUtils.getOpCode(instString);
		return ( opcode.equalsIgnoreCase("valuepick"));
	}*/
	

}
