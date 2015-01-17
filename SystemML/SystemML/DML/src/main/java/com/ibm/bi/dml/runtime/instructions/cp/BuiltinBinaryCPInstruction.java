/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.cp;

import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.functionobjects.Builtin;
import com.ibm.bi.dml.runtime.functionobjects.ValueFunction;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.matrix.operators.BinaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.matrix.operators.RightScalarOperator;


public abstract class BuiltinBinaryCPInstruction extends BinaryCPInstruction 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private int arity;
	
	public BuiltinBinaryCPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out, int _arity, String opcode, String istr )
	{
		super(op, in1, in2, out, opcode, istr);
		_cptype = CPINSTRUCTION_TYPE.BuiltinBinary;
		arity = _arity;
	}

	public int getArity() {
		return arity;
	}
	
	public static Instruction parseInstruction ( String str ) 
		throws DMLRuntimeException, DMLUnsupportedOperationException {
		CPOperand in1 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand in2 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		String opcode = parseBinaryInstruction(str, in1, in2, out);
		
		ValueFunction func = Builtin.getBuiltinFnObject(opcode);
		
		// Determine appropriate Function Object based on opcode
			
		if ( in1.getDataType() == DataType.SCALAR && in2.getDataType() == DataType.SCALAR ) {
			return new ScalarScalarBuiltinCPInstruction(new BinaryOperator(func), in1, in2, out, opcode, str);
		} else if (in1.getDataType() != in2.getDataType()) {
			return new MatrixScalarBuiltinCPInstruction(new RightScalarOperator(func, 0), in1, in2, out, opcode, str);					
		} else { // if ( in1.getDataType() == DataType.MATRIX && in2.getDataType() == DataType.MATRIX ) {
			return new MatrixMatrixBuiltinCPInstruction(new BinaryOperator(func), in1, in2, out, opcode, str);	
		} 
	}
}
