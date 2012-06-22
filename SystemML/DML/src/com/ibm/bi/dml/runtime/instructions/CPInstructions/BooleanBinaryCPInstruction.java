package com.ibm.bi.dml.runtime.instructions.CPInstructions;

import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.matrix.operators.BinaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.utils.DMLRuntimeException;


public class BooleanBinaryCPInstruction extends BinaryCPInstruction {
	public BooleanBinaryCPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out, String istr )
	{
		super(op, in1, in2, out, istr);
		cptype = CPINSTRUCTION_TYPE.BooleanBinary;
	}
	
	public static Instruction parseInstruction (String str) throws DMLRuntimeException {
		CPOperand in1 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand in2 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		String opcode = parseBinaryInstruction(str, in1, in2, out);
		
		// Boolean operations must be performed on BOOLEAN
		ValueType vt1, vt2, vt3;
		vt1 = vt2 = vt3 = null;
		vt1 = in1.get_valueType();
		if ( in2 != null )
			vt2 = in2.get_valueType();
		vt3 = out.get_valueType();
		if ( vt1 != ValueType.BOOLEAN || vt3 != ValueType.BOOLEAN 
				|| (vt2 != null && vt2 != ValueType.BOOLEAN) )
			throw new DMLRuntimeException("Unexpected ValueType in ArithmeticInstruction.");
		
		
		// Determine appropriate Function Object based on opcode	
		return new BooleanBinaryCPInstruction(getBinaryOperator(opcode), in1, in2, out, str);
	}
	
	@Override
	public void processInstruction(ProgramBlock pb) throws DMLRuntimeException {
		ScalarObject so1 = pb.getScalarInput(input1.get_name(), input1.get_valueType());
		ScalarObject so2 = null;
		if ( input2 != null ) 
			so2 = pb.getScalarInput(input2.get_name(), input2.get_valueType() );
		ScalarObject sores = null;
		
		BinaryOperator dop = (BinaryOperator) optr;
		boolean rval = dop.fn.execute(so1.getBooleanValue(), so2.getBooleanValue());
		sores = (ScalarObject) new BooleanObject(rval);
		
		pb.setScalarOutput(output.get_name(), sores);
	}
}
