package com.ibm.bi.dml.runtime.instructions.spark;

import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.functionobjects.Builtin;
import com.ibm.bi.dml.runtime.functionobjects.ValueFunction;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.cp.CPOperand;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.matrix.operators.UnaryOperator;


public abstract class BuiltinUnarySPInstruction extends UnarySPInstruction 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public BuiltinUnarySPInstruction(Operator op, CPOperand in, CPOperand out, String opcode, String istr )
	{
		super(op, in, out, opcode, istr);
		_sptype = SPINSTRUCTION_TYPE.BuiltinUnary;
	}
	
	/**
	 * 
	 * @param str
	 * @return
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException
	 */
	public static Instruction parseInstruction ( String str ) 
			throws DMLRuntimeException, DMLUnsupportedOperationException 
	{
		CPOperand in = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		
		String opcode = parseUnaryInstruction(str, in, out);
		ValueFunction func = Builtin.getBuiltinFnObject(opcode);
		return new MatrixBuiltinSPInstruction(new UnaryOperator(func), in, out, opcode, str);
	}
}
