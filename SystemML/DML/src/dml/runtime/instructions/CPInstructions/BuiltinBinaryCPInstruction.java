package dml.runtime.instructions.CPInstructions;

import dml.parser.Expression.DataType;
import dml.parser.Expression.ValueType;
import dml.runtime.functionobjects.Builtin;
import dml.runtime.functionobjects.ValueFunction;
import dml.runtime.instructions.Instruction;
import dml.runtime.matrix.operators.BinaryOperator;
import dml.runtime.matrix.operators.Operator;
import dml.runtime.matrix.operators.SimpleOperator;
import dml.runtime.matrix.operators.RightScalarOperator;
import dml.utils.DMLRuntimeException;
import dml.utils.DMLUnsupportedOperationException;

public class BuiltinBinaryCPInstruction extends BinaryCPInstruction {
	int arity;
	
	public BuiltinBinaryCPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out, int _arity, String istr )
	{
		super(op, in1, in2, out, istr);
		cptype = CPINSTRUCTION_TYPE.BuiltinBinary;
		instString = istr;
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
		
		/*
		int _arity = parts.length - 2;
		
		boolean b = ((Builtin) func).checkArity(_arity);
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
			
		if ( in1.get_dataType() == DataType.SCALAR && in2.get_dataType() == DataType.SCALAR ) {
			return new ScalarScalarBuiltinCPInstruction(new BinaryOperator(func), in1, in2, out, str);
		} else if (in1.get_dataType() != in2.get_dataType()) {
			return new MatrixScalarBuiltinCPInstruction(new RightScalarOperator(func, 0), in1, in2, out, str);					
		} else { // if ( in1.get_dataType() == DataType.MATRIX && in2.get_dataType() == DataType.MATRIX ) {
			return new MatrixMatrixBuiltinCPInstruction(new BinaryOperator(func), in1, in2, out, str);	
		} 
	}
}
