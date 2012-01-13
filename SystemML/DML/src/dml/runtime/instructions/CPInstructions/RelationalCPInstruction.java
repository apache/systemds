package dml.runtime.instructions.CPInstructions;

import dml.parser.Expression.ValueType;
import dml.runtime.controlprogram.ProgramBlock;
import dml.runtime.functionobjects.Equals;
import dml.runtime.functionobjects.GreaterThan;
import dml.runtime.functionobjects.GreaterThanEquals;
import dml.runtime.functionobjects.LessThan;
import dml.runtime.functionobjects.LessThanEquals;
import dml.runtime.functionobjects.NotEquals;
import dml.runtime.instructions.Instruction;
import dml.runtime.instructions.InstructionUtils;
import dml.runtime.matrix.operators.Operator;
import dml.runtime.matrix.operators.SimpleOperator;
import dml.utils.DMLRuntimeException;

public class RelationalCPInstruction extends ScalarCPInstruction {

	public RelationalCPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out, int _arity, String istr )
	{
		super(op, in1, in2, out);
		cptype = CPINSTRUCTION_TYPE.Relational;
		instString = istr;
	}
	
	public static Instruction parseInstruction ( String str ) throws DMLRuntimeException {
		
		String opcode = InstructionUtils.getOpCode(str);
		
		int _arity = 2;
		InstructionUtils.checkNumFields ( str, _arity + 1 ); // + 1 for the output
		
		String[] parts = InstructionUtils.getInstructionPartsWithValueType ( str );
		
		CPOperand in1, in2, out;
		opcode = parts[0];
		in1 = new CPOperand(parts[1]);
		in2 = new CPOperand(parts[2]);
		out = new CPOperand(parts[3]);
		
		// TODO: Relational operations need not have value type checking
		ValueType vt1, vt3;
		vt1 = vt3 = null;
		vt1 = in1.get_valueType();
		vt3 = out.get_valueType();
		if ( vt3 != ValueType.BOOLEAN )
			throw new DMLRuntimeException("Unexpected ValueType in RelationalCPInstruction: " + str);
		
		if ( vt1 == ValueType.BOOLEAN && !opcode.equalsIgnoreCase("==") && !opcode.equalsIgnoreCase("!=") ) 
			throw new DMLRuntimeException("Operation " + opcode + " can not be applied on boolean values (Instruction = " + str + ").");
		
		// Determine appropriate Function Object based on opcode
		
		if ( opcode.equalsIgnoreCase("==") ) {
			return new RelationalCPInstruction(new SimpleOperator(Equals.getEqualsFnObject()), in1, in2, out, _arity, str);
		} 
		else if ( opcode.equalsIgnoreCase("!=") ) {
			return new RelationalCPInstruction(new SimpleOperator(NotEquals.getNotEqualsFnObject()), in1, in2, out, _arity, str);
		}
		else if ( opcode.equalsIgnoreCase("<") ) {
			return new RelationalCPInstruction(new SimpleOperator(LessThan.getLessThanFnObject()), in1, in2, out, _arity, str);
		}
		else if ( opcode.equalsIgnoreCase(">") ) {
			return new RelationalCPInstruction(new SimpleOperator(GreaterThan.getGreaterThanFnObject()), in1, in2, out, _arity, str);
		}
		else if ( opcode.equalsIgnoreCase("<=") ) {
			return new RelationalCPInstruction(new SimpleOperator(LessThanEquals.getLessThanEqualsFnObject()), in1, in2, out, _arity, str);
		}
		else if ( opcode.equalsIgnoreCase(">=") ) {
			return new RelationalCPInstruction(new SimpleOperator(GreaterThanEquals.getGreaterThanEqualsFnObject()), in1, in2, out, _arity, str);
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
		
		if ( input1.get_valueType() == ValueType.INT && input2.get_valueType() == ValueType.INT ) {
			boolean rval = dop.fn.compare ( so1.getIntValue(), so2.getIntValue() );
			sores = (ScalarObject) new BooleanObject(rval); 
		}
		
		else if ( input1.get_valueType() == ValueType.DOUBLE && input2.get_valueType() == ValueType.DOUBLE ) {
			boolean rval = dop.fn.compare ( so1.getDoubleValue(), so2.getDoubleValue() );
			sores = (ScalarObject) new BooleanObject(rval); 
		}
		
		else if ( input1.get_valueType() == ValueType.INT && input2.get_valueType() == ValueType.DOUBLE ) {
			boolean rval = dop.fn.compare ( so1.getIntValue(), so2.getDoubleValue() );
			sores = (ScalarObject) new BooleanObject(rval); 
		}
		
		else if ( input1.get_valueType() == ValueType.DOUBLE && input2.get_valueType() == ValueType.INT ) {
			boolean rval = dop.fn.compare ( so1.getDoubleValue(), so2.getIntValue() );
			sores = (ScalarObject) new BooleanObject(rval); 
		}
		
		else if ( input1.get_valueType() == ValueType.BOOLEAN && input2.get_valueType() == ValueType.BOOLEAN ) {
			boolean rval = dop.fn.compare ( so1.getBooleanValue(), so2.getBooleanValue() );
			sores = (ScalarObject) new BooleanObject(rval); 
		}
		
		else {
			throw new DMLRuntimeException("compare(): Invalid combination of value types.");
		}
		
		// 3) Put the result value into ProgramBlock
		pb.setVariable(output.get_name(), sores);
		return sores;
		
	}
}
