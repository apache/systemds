package com.ibm.bi.dml.runtime.instructions.CPInstructions;

import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.functionobjects.Builtin;
import com.ibm.bi.dml.runtime.functionobjects.ValueFunction;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.matrix.operators.SimpleOperator;
import com.ibm.bi.dml.runtime.matrix.operators.UnaryOperator;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.DMLUnsupportedOperationException;


public class BuiltinUnaryCPInstruction extends UnaryCPInstruction {
	int arity;
	
	public BuiltinUnaryCPInstruction(Operator op, CPOperand in, CPOperand out, int _arity, String istr )
	{
		super(op, in, out, istr);
		cptype = CPINSTRUCTION_TYPE.BuiltinUnary;
		instString = istr;
		arity = _arity;
	}

	public int getArity() {
		return arity;
	}
	
	public static Instruction parseInstruction ( String str ) 
		throws DMLRuntimeException, DMLUnsupportedOperationException {
		CPOperand in = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		String opcode = parseUnaryInstruction(str, in, out);
		
		ValueFunction func = Builtin.getBuiltinFnObject(opcode);
		
		/*
	    int _arity = parts.length - 2;
		boolean b = ((Builtin) func).checkArity(arity);
		if ( !b ) {
			throw new DMLRuntimeException("Invalid number of inputs to builtin function: " 
										  + ((Builtin) func).bFunc);
		}
		*/
		
		// TODO: VALUE TYPE CHECKING
		/*
		ValueType vt1, vt2, vt3;
		vt1 = vt2 = vt3 = null;
		vt1 = in1.get_valueType();
		if ( in2 != null )
			vt2 = in2.get_valueType();
		vt3 = out.get_valueType();
		if ( vt1 != ValueType.BOOLEAN || vt3 != ValueType.BOOLEAN 
				|| (vt2 != null && vt2 != ValueType.BOOLEAN) )
			throw new DMLRuntimeException("Unexpected ValueType in ArithmeticInstruction.");
		*/
		
		// Determine appropriate Function Object based on opcode
			
		if(in.get_dataType() == DataType.SCALAR)
			return new ScalarBuiltinCPInstruction(new SimpleOperator(func), in, out, str);
		else if(in.get_dataType() == DataType.MATRIX)
			return new MatrixBuiltinCPInstruction(new UnaryOperator(func), in, out, str);
		
		return null;
	}
}
