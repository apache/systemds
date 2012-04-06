package dml.runtime.instructions.CPInstructions;

import dml.runtime.controlprogram.ProgramBlock;
import dml.runtime.instructions.InstructionUtils;
import dml.runtime.matrix.operators.SimpleOperator;
import dml.runtime.matrix.operators.Operator;
import dml.utils.DMLRuntimeException;

public class ScalarBuiltinCPInstruction extends BuiltinUnaryCPInstruction{
	public ScalarBuiltinCPInstruction(Operator op,
									  CPOperand in,
									  CPOperand out,
									  String instr){
		super(op, in, out, 1, instr);
	}

	@Override 
	public ScalarObject processInstruction(ProgramBlock pb) 
		throws DMLRuntimeException {
		
		String opcode = InstructionUtils.getOpCode(instString);
		
		ScalarObject so = pb.getScalarVariable( input1.get_name(), input1.get_valueType() );
		
		ScalarObject sores = null;
		
		SimpleOperator dop = (SimpleOperator) optr;
			
		if ( opcode.equalsIgnoreCase("print") ) {
			switch (input1.get_valueType()) {
			case INT:
				System.out.println("" + so.getIntValue());
				break;
			case DOUBLE:
				System.out.println("" + so.getDoubleValue());
				break;
			case BOOLEAN:
				System.out.println("" + so.getBooleanValue());
			break;
			case STRING:
				System.out.println("" + so.getStringValue());
				break;
			}
		}
		else if (opcode.equalsIgnoreCase("print2")) {
			System.out.println(so.getStringValue());
		}
		else {
			/*
			 * Inputs for all builtins other than PRINT are treated as DOUBLE.
			 */
			double rval;
			rval = dop.fn.execute(so.getDoubleValue());
			sores = (ScalarObject) new DoubleObject(rval);
		}
		
		//prithvi TODO: we input a null into the symbol table
		//if builtin is print/print2?? is that ok?
		pb.setVariable(output.get_name(), sores);
		return sores;
	}
}
