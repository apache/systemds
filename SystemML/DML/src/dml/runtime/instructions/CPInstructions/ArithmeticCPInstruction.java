package dml.runtime.instructions.CPInstructions;

import dml.parser.Expression.ValueType;
import dml.runtime.controlprogram.ProgramBlock;
import dml.runtime.functionobjects.Divide;
import dml.runtime.functionobjects.Minus;
import dml.runtime.functionobjects.Multiply;
import dml.runtime.functionobjects.Plus;
import dml.runtime.functionobjects.Power;
import dml.runtime.instructions.Instruction;
import dml.runtime.instructions.InstructionUtils;
import dml.runtime.matrix.operators.Operator;
import dml.runtime.matrix.operators.SimpleOperator;
import dml.utils.DMLRuntimeException;

public class ArithmeticCPInstruction extends ScalarCPInstruction {

	public ArithmeticCPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out, String istr )
	{
		super(op, in1, in2, out);
		cptype = CPINSTRUCTION_TYPE.Arithmetic;
		instString = istr;
	}
	
	public static Instruction parseInstruction ( String str ) throws DMLRuntimeException {
		
		InstructionUtils.checkNumFields ( str, 3 );
		
		String[] parts = InstructionUtils.getInstructionPartsWithValueType ( str );
		
		CPOperand in1, in2, out;
		String opcode = parts[0];
		in1 = new CPOperand(parts[1]);
		in2 = new CPOperand(parts[2]);
		out = new CPOperand(parts[3]);
		
		// Arithmetic operations must be performed on DOUBLE or INT
		ValueType vt1 = in1.get_valueType();
		ValueType vt2 = in2.get_valueType();
		ValueType vt3 = out.get_valueType();
		
		// Determine appropriate Function Object based on opcode
		
		if ( opcode.equalsIgnoreCase("+") ) {
			//if ( vt1 != ValueType.DOUBLE && vt1 != ValueType.INT && vt1 != ValueType.STRING )
			//	throw new DMLRuntimeException("Unexpected ValueType (" + vt1 + ") in ArithmeticInstruction: " + str);
			return new ArithmeticCPInstruction(new SimpleOperator(Plus.getPlusFnObject()), in1, in2, out, str);
		} 
		else {
			if ( (vt1 != ValueType.DOUBLE && vt1 != ValueType.INT)
					|| (vt2 != ValueType.DOUBLE && vt2 != ValueType.INT)
					|| (vt3 != ValueType.DOUBLE && vt3 != ValueType.INT) )
				throw new DMLRuntimeException("Unexpected ValueType in ArithmeticInstruction.");
			
			if ( vt1 != ValueType.DOUBLE && vt1 != ValueType.INT ) {
				throw new DMLRuntimeException("Unexpected ValueType (" + vt1 + ") in ArithmeticInstruction: " + str);
			}
			
			if ( opcode.equalsIgnoreCase("-") ) {
				return new ArithmeticCPInstruction(new SimpleOperator(Minus.getMinusFnObject()), in1, in2, out, str);
			}
			else if ( opcode.equalsIgnoreCase("*") ) {
				return new ArithmeticCPInstruction(new SimpleOperator(Multiply.getMultiplyFnObject()), in1, in2, out, str);
			}
			else if ( opcode.equalsIgnoreCase("/") ) {
				return new ArithmeticCPInstruction(new SimpleOperator(Divide.getDivideFnObject()), in1, in2, out, str);
			}
			else if ( opcode.equalsIgnoreCase("^") ) {
				return new ArithmeticCPInstruction(new SimpleOperator(Power.getPowerFnObject()), in1, in2, out, str);
			}
		}
		return null;
	}

	@Override
	public ScalarObject processInstruction( ProgramBlock pb ) throws DMLRuntimeException {
		
		// 1) Obtain data objects associated with inputs 
		ScalarObject so1 = pb.getScalarVariable(input1.get_name(), input1.get_valueType());
		ScalarObject so2 = pb.getScalarVariable(input2.get_name(), input2.get_valueType() );
		ScalarObject sores = null;
		
		// 2) Compute the result value & make an appropriate data object 
		SimpleOperator dop = (SimpleOperator) optr;
		
		if ( input1.get_valueType() == ValueType.STRING || input2.get_valueType() == ValueType.STRING ) {
			String rval = dop.fn.execute(so1.getStringValue(), so2.getStringValue());
			sores = (ScalarObject) new StringObject(rval);
		}
		else if ( input1.get_valueType() == ValueType.INT && input2.get_valueType() == ValueType.INT ) {
			// TODO: statiko: can we remove the use of instanceof ?
			if ( dop.fn instanceof Divide || dop.fn instanceof Power ) {
				// If both inputs are of type INT then output must be an INT if operation is not divide or power
				double rval = dop.fn.execute ( so1.getIntValue(), so2.getIntValue() );
				sores = (ScalarObject) new DoubleObject(rval);
			}
			else {
				// If both inputs are of type INT then output must be an INT if operation is not divide or power
				int rval = (int) dop.fn.execute ( so1.getIntValue(), so2.getIntValue() );
				sores = (ScalarObject) new IntObject(rval);
			}
		}
		
		else {
			// If either of the input is of type DOUBLE then output is a DOUBLE
			double rval = dop.fn.execute ( so1.getDoubleValue(), so2.getDoubleValue() );
			sores = (ScalarObject) new DoubleObject(rval); 
		}
		
		// 3) Put the result value into ProgramBlock
		pb.setVariable(output.get_name(), sores);
		return sores;
		
	}
	
}
