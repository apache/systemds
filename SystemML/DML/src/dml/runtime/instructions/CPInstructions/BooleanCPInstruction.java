package dml.runtime.instructions.CPInstructions;

import dml.parser.Expression.ValueType;
import dml.runtime.controlprogram.ProgramBlock;
import dml.runtime.functionobjects.And;
import dml.runtime.functionobjects.Not;
import dml.runtime.functionobjects.Or;
import dml.runtime.instructions.Instruction;
import dml.runtime.instructions.InstructionUtils;
import dml.runtime.matrix.operators.Operator;
import dml.runtime.matrix.operators.SimpleOperator;
import dml.utils.DMLRuntimeException;

public class BooleanCPInstruction extends ScalarCPInstruction {
	int arity;
	
	public BooleanCPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out, int _arity, String istr )
	{
		super(op, in1, in2, out);
		cptype = CPINSTRUCTION_TYPE.Boolean;
		arity = _arity;
		instString = istr;
	}
	
	public static Instruction parseInstruction ( String str ) throws DMLRuntimeException {
		
		String opcode = InstructionUtils.getOpCode(str);
		
		int _arity = 2;
		if ( opcode.equalsIgnoreCase("!") )
			_arity = 1;
		
		InstructionUtils.checkNumFields ( str, _arity+1 ); //+1 for output
		
		String[] parts = InstructionUtils.getInstructionPartsWithValueType ( str );
		
		CPOperand in1, in2, out;
		opcode = parts[0];
		in1 = new CPOperand(parts[1]);
		if ( _arity == 1 ) {
			in2 = null;
			out = new CPOperand(parts[2]);
		}
		else {
			in2 = new CPOperand(parts[2]);
			out = new CPOperand(parts[3]);
		}
		
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
		
		if ( opcode.equalsIgnoreCase("&&") ) {
			return new BooleanCPInstruction(new SimpleOperator(And.getAndFnObject()), in1, in2, out, _arity, str);
		} 
		else if ( opcode.equalsIgnoreCase("||") ) {
			return new BooleanCPInstruction(new SimpleOperator(Or.getOrFnObject()), in1, in2, out, _arity, str);
		}
		else if ( opcode.equalsIgnoreCase("!") ) {
			return new BooleanCPInstruction(new SimpleOperator(Not.getNotFnObject()), in1, in2, out, _arity, str);
		}
		return null;
	}
	
	@Override
	public ScalarObject processInstruction(ProgramBlock pb) throws DMLRuntimeException {
		// 1) Obtain data objects associated with inputs 
		ScalarObject so1 = pb.getScalarVariable(input1.get_name(), input1.get_valueType());
		ScalarObject so2 = null;
		if ( input2 != null ) 
			so2 = pb.getScalarVariable(input2.get_name(), input2.get_valueType() );
		ScalarObject sores = null;
		
		// 2) Compute the result value & make an appropriate data object 
		SimpleOperator dop = (SimpleOperator) optr;
		boolean rval;
		if ( arity == 2 ) {
			rval = dop.fn.execute(so1.getBooleanValue(), so2.getBooleanValue());
		} 
		else {
			rval = dop.fn.execute(so1.getBooleanValue());
		}
		sores = (ScalarObject) new BooleanObject(rval);
		
		// 3) Put the result value into ProgramBlock
		pb.setVariable(output.get_name(), sores);
		return sores;
	}
}
