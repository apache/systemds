package com.ibm.bi.dml.runtime.instructions.CPInstructions;

import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.matrix.operators.SimpleOperator;
import com.ibm.bi.dml.utils.DMLRuntimeException;


public class ScalarBuiltinCPInstruction extends BuiltinUnaryCPInstruction{
	public ScalarBuiltinCPInstruction(Operator op,
									  CPOperand in,
									  CPOperand out,
									  String instr){
		super(op, in, out, 1, instr);
	}

	@Override 
	public void processInstruction(ProgramBlock pb) 
		throws DMLRuntimeException {
		
		String opcode = InstructionUtils.getOpCode(instString);
		
		ScalarObject so = pb.getScalarInput( input1.get_name(), input1.get_valueType() );
		
		ScalarObject sores = null;
		
		SimpleOperator dop = (SimpleOperator) optr;
			
		if ( opcode.equalsIgnoreCase("print") ) {
			String outString = "";
			switch (input1.get_valueType()) {
			case INT:
				outString += so.getIntValue();
				break;
			case DOUBLE:
				outString += so.getDoubleValue();
				break;
			case BOOLEAN:
				outString += so.getBooleanValue();
				break;
			case STRING:
				outString += so.getStringValue();
				break;
			}
			System.out.println(outString);
			// String that is printed on stdout will be inserted into symbol table (dummy, not necessary!) 
			sores = new StringObject(outString);
		}
		else if (opcode.equalsIgnoreCase("print2")) {
			System.out.println(so.getStringValue());
			// String that is printed on stdout will be inserted into symbol table (dummy, not necessary!) 
			sores = new StringObject(so.getStringValue());
		}
		else {
			/*
			 * Inputs for all builtins other than PRINT are treated as DOUBLE.
			 */
			double rval;
			rval = dop.fn.execute(so.getDoubleValue());
			sores = (ScalarObject) new DoubleObject(rval);
		}
		
		pb.setScalarOutput(output.get_name(), sores);
	}
}
