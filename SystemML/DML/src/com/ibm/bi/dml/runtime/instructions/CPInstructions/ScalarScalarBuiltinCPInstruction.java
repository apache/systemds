package com.ibm.bi.dml.runtime.instructions.CPInstructions;

import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.operators.BinaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.utils.DMLRuntimeException;


public class ScalarScalarBuiltinCPInstruction extends BuiltinBinaryCPInstruction{
	public ScalarScalarBuiltinCPInstruction(Operator op,
			  								CPOperand in1,
			  								CPOperand in2,
			  								CPOperand out,
			  								String instr){
		super(op, in1, in2, out, 2, instr);
	}

	@Override 
	public void processInstruction(ProgramBlock pb) throws DMLRuntimeException {
		
		String opcode = InstructionUtils.getOpCode(instString);
		ScalarObject sores = null;
		
		ScalarObject so1 = pb.getScalarInput( input1.get_name(), input1.get_valueType() );
		ScalarObject so2 = pb.getScalarInput(input2.get_name(), input2.get_valueType() );
		
		if ( opcode.equalsIgnoreCase("print") ) {
			String buffer = "";
			if (input2.get_valueType() != ValueType.STRING)
				throw new DMLRuntimeException("wrong value type in print");
			buffer = so2.getStringValue() + " ";
					
			switch (input1.get_valueType()) {
			case INT:
				System.out.println(buffer + so1.getIntValue());
				break;
			case DOUBLE:
				System.out.println(buffer + so1.getDoubleValue());
				break;
			case BOOLEAN:
				System.out.println(buffer + so1.getBooleanValue());
				break;
			case STRING:
				System.out.println(buffer + so1.getStringValue());
				break;
			}
		}
		else {
			/*
			 * Inputs for all builtins other than PRINT are treated as DOUBLE.
			 */
			BinaryOperator dop = (BinaryOperator) optr;
			double rval = dop.fn.execute(so1.getDoubleValue(), so2.getDoubleValue());
			
			sores = (ScalarObject) new DoubleObject(rval);
		}
		
		// 3) Put the result value into ProgramBlock
		pb.setScalarOutput(output.get_name(), sores);
	}
}
