package com.ibm.bi.dml.runtime.instructions.CPInstructions;

import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.matrix.operators.SimpleOperator;
import com.ibm.bi.dml.utils.DMLRuntimeException;


public class BooleanUnaryCPInstruction extends UnaryCPInstruction{
	public BooleanUnaryCPInstruction(Operator op,
									 CPOperand in,
									 CPOperand out,
									 String instr){
		super(op, in, out, instr);
		cptype = CPINSTRUCTION_TYPE.BooleanUnary;
	}

	public static Instruction parseInstruction (String str) throws DMLRuntimeException {
		CPOperand in = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		String opcode = parseUnaryInstruction(str, in, out);
		
		// Boolean operations must be performed on BOOLEAN
		ValueType vt1, vt2;
		vt1 = vt2 = null;
		vt1 = in.get_valueType();
		vt2 = out.get_valueType();
		if ( vt1 != ValueType.BOOLEAN || vt2 != ValueType.BOOLEAN )
			throw new DMLRuntimeException("Unexpected ValueType in ArithmeticInstruction.");
		
		// Determine appropriate Function Object based on opcode	
		return new BooleanUnaryCPInstruction(getSimpleUnaryOperator(opcode), in, out, str);
	}
	
	@Override
	public ScalarObject processInstruction(ProgramBlock pb) throws DMLRuntimeException {
		// 1) Obtain data objects associated with inputs 
		ScalarObject so = pb.getScalarVariable(input1.get_name(), input1.get_valueType());
		ScalarObject sores = null;
		
		// 2) Compute the result value & make an appropriate data object 
		SimpleOperator dop = (SimpleOperator) optr;
		boolean rval;
		rval = dop.fn.execute(so.getBooleanValue());
		
		sores = (ScalarObject) new BooleanObject(rval);
		
		// 3) Put the result value into ProgramBlock
		pb.setVariable(output.get_name(), sores);
		return sores;
	}
}
