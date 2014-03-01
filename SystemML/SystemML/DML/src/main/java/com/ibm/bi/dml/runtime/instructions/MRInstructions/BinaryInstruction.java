/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.MRInstructions;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.functionobjects.Builtin;
import com.ibm.bi.dml.runtime.functionobjects.Divide;
import com.ibm.bi.dml.runtime.functionobjects.EqualsReturnDouble;
import com.ibm.bi.dml.runtime.functionobjects.GreaterThanEqualsReturnDouble;
import com.ibm.bi.dml.runtime.functionobjects.GreaterThanReturnDouble;
import com.ibm.bi.dml.runtime.functionobjects.IntegerDivide;
import com.ibm.bi.dml.runtime.functionobjects.LessThanEqualsReturnDouble;
import com.ibm.bi.dml.runtime.functionobjects.LessThanReturnDouble;
import com.ibm.bi.dml.runtime.functionobjects.Minus;
import com.ibm.bi.dml.runtime.functionobjects.Modulus;
import com.ibm.bi.dml.runtime.functionobjects.Multiply;
import com.ibm.bi.dml.runtime.functionobjects.NotEqualsReturnDouble;
import com.ibm.bi.dml.runtime.functionobjects.Plus;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.io.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.io.OperationsOnMatrixValues;
import com.ibm.bi.dml.runtime.matrix.mapred.CachedValueMap;
import com.ibm.bi.dml.runtime.matrix.mapred.IndexedMatrixValue;
import com.ibm.bi.dml.runtime.matrix.operators.BinaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;


public class BinaryInstruction extends BinaryMRInstructionBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public BinaryInstruction(Operator op, byte in1, byte in2, byte out, String istr)
	{
		super(op, in1, in2, out);
		mrtype = MRINSTRUCTION_TYPE.ArithmeticBinary;
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
		
		if ( opcode.equalsIgnoreCase("+") ) {
			return new BinaryInstruction(new BinaryOperator(Plus.getPlusFnObject()), in1, in2, out, str);
		} 
		else if ( opcode.equalsIgnoreCase("-") ) {
			return new BinaryInstruction(new BinaryOperator(Minus.getMinusFnObject()), in1, in2, out, str);
		}
		else if ( opcode.equalsIgnoreCase("*") ) {
			return new BinaryInstruction(new BinaryOperator(Multiply.getMultiplyFnObject()), in1, in2, out, str);
		}
		else if ( opcode.equalsIgnoreCase("/") ) {
			return new BinaryInstruction(new BinaryOperator(Divide.getDivideFnObject()), in1, in2, out, str);
		}
		else if ( opcode.equalsIgnoreCase("%%") ) {
			return new BinaryInstruction(new BinaryOperator(Modulus.getModulusFnObject()), in1, in2, out, str);
		}
		else if ( opcode.equalsIgnoreCase("%/%") ) {
			return new BinaryInstruction(new BinaryOperator(IntegerDivide.getIntegerDivideFnObject()), in1, in2, out, str);
		}
		else if ( opcode.equalsIgnoreCase("max") ) {
			return new BinaryInstruction(new BinaryOperator(Builtin.getBuiltinFnObject("max")), in1, in2, out, str);
		}
		else if ( opcode.equalsIgnoreCase(">") ) {
			return new BinaryInstruction(new BinaryOperator(GreaterThanReturnDouble.getGreaterThanReturnDoubleFnObject()), in1, in2, out, str);
		}
		else if ( opcode.equalsIgnoreCase(">=") ) {
			return new BinaryInstruction(new BinaryOperator(GreaterThanEqualsReturnDouble.getGreaterThanEqualsReturnDoubleFnObject()), in1, in2, out, str);
		}
		else if ( opcode.equalsIgnoreCase("<") ) {
			return new BinaryInstruction(new BinaryOperator(LessThanReturnDouble.getLessThanReturnDoubleFnObject()), in1, in2, out, str);
		}
		else if ( opcode.equalsIgnoreCase("<=") ) {
			return new BinaryInstruction(new BinaryOperator(LessThanEqualsReturnDouble.getLessThanEqualsReturnDoubleFnObject()), in1, in2, out, str);
		}
		else if ( opcode.equalsIgnoreCase("==") ) {
			return new BinaryInstruction(new BinaryOperator(EqualsReturnDouble.getEqualsReturnDoubleFnObject()), in1, in2, out, str);
		}
		else if ( opcode.equalsIgnoreCase("!=") ) {
			return new BinaryInstruction(new BinaryOperator(NotEqualsReturnDouble.getNotEqualsReturnDoubleFnObject()), in1, in2, out, str);
		}
		return null;
	}
	
	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass,
			CachedValueMap cachedValues, IndexedMatrixValue tempValue, IndexedMatrixValue zeroInput,
			int blockRowFactor, int blockColFactor)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		
		IndexedMatrixValue in1=cachedValues.getFirst(input1);
		IndexedMatrixValue in2=cachedValues.getFirst(input2);
		if(in1==null && in2==null)
			return;
		
		//allocate space for the output value
		//try to avoid coping as much as possible
		IndexedMatrixValue out;
		if( (output!=input1 && output!=input2)
			|| (output==input1 && in1==null)
			|| (output==input2 && in2==null) )
			out=cachedValues.holdPlace(output, valueClass);
		else
			out=tempValue;
		
		//if one of the inputs is null, then it is a all zero block
		MatrixIndexes finalIndexes=null;
		if(in1==null)
		{
			in1=zeroInput;
			in1.getValue().reset(in2.getValue().getNumRows(), 
					in2.getValue().getNumColumns());
			finalIndexes=in2.getIndexes();
		}else
			finalIndexes=in1.getIndexes();
		
		if(in2==null)
		{
			in2=zeroInput;
			in2.getValue().reset(in1.getValue().getNumRows(), 
					in1.getValue().getNumColumns());
		}
		
		//process instruction
		out.getIndexes().setIndexes(finalIndexes);
		OperationsOnMatrixValues.performBinaryIgnoreIndexes(in1.getValue(), 
				in2.getValue(), out.getValue(), ((BinaryOperator)optr));
		
		//put the output value in the cache
		if(out==tempValue)
			cachedValues.add(output, out);
		
	}

}
