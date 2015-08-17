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
