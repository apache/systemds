package org.apache.sysml.runtime.instructions.spark;

import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.DMLUnsupportedOperationException;
import org.apache.sysml.runtime.functionobjects.Builtin;
import org.apache.sysml.runtime.functionobjects.ValueFunction;
import org.apache.sysml.runtime.instructions.Instruction;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.matrix.operators.Operator;
import org.apache.sysml.runtime.matrix.operators.UnaryOperator;


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
