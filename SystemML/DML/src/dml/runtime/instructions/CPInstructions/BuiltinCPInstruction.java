package dml.runtime.instructions.CPInstructions;

import dml.parser.Expression.ValueType;
import dml.runtime.controlprogram.ProgramBlock;
import dml.runtime.functionobjects.Builtin;
import dml.runtime.functionobjects.ValueFunction;
import dml.runtime.instructions.Instruction;
import dml.runtime.instructions.InstructionUtils;
import dml.runtime.matrix.MatrixCharacteristics;
import dml.runtime.matrix.MatrixDimensionsMetaData;
import dml.runtime.matrix.operators.Operator;
import dml.runtime.matrix.operators.SimpleOperator;
import dml.utils.DMLRuntimeException;
import dml.utils.DMLUnsupportedOperationException;

public class BuiltinCPInstruction extends ScalarCPInstruction {
	int arity;
	public BuiltinCPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out, int _arity, String istr )
	{
		super(op, in1, in2, out);
		cptype = CPINSTRUCTION_TYPE.Builtin;
		arity = _arity;
		instString = istr;
	}

	public int getArity() {
		return arity;
	}
	
	public static Instruction parseInstruction ( String str ) throws DMLRuntimeException, DMLUnsupportedOperationException {
		
		String opcode = InstructionUtils.getOpCode(str);
		ValueFunction func = Builtin.getBuiltinFnObject(opcode);
		
		// InstructionUtils.checkNumInputs ( str, _arity );
		
		String[] parts = InstructionUtils.getInstructionPartsWithValueType ( str );
		int _arity = parts.length - 2;
		
		boolean b = ((Builtin) func).checkArity(_arity);
		if ( !b ) {
			throw new DMLRuntimeException("Invalid number of inputs to builtin function: " + ((Builtin) func).bFunc);
		}
		
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
		
		return new BuiltinCPInstruction(new SimpleOperator(func), in1, in2, out, _arity, str);
	}
	
	@Override 
	public ScalarObject processInstruction(ProgramBlock pb) throws DMLRuntimeException {
		
		// Handle NROW, NCOL, LENGTH builtins that do not have any input scalar variables
		String opcode = InstructionUtils.getOpCode(instString);
		ScalarObject sores = null;
		
		
		MatrixCharacteristics matchar = null;
		/*
		 * For nrow, ncol, length 
		 *   -- first operand is a string denoting the filename
		 *   -- second operand will capture the result of builtin
		 */
		if ( opcode.equalsIgnoreCase("nrow")) {
			matchar = ((MatrixDimensionsMetaData)pb.getMetaData(input1.get_name())).getMatrixCharacteristics();
			sores = new DoubleObject(matchar.numRows);
		} 
		else if ( opcode.equalsIgnoreCase("ncol")) {
			matchar = ((MatrixDimensionsMetaData)pb.getMetaData(input1.get_name())).getMatrixCharacteristics();
			sores = new DoubleObject(matchar.numColumns);			
		}
		else if ( opcode.equalsIgnoreCase("length")) {
			matchar = ((MatrixDimensionsMetaData)pb.getMetaData(input1.get_name())).getMatrixCharacteristics();
			sores = new DoubleObject(matchar.numRows * matchar.numColumns);
		}
		else {
			// 1) Obtain data objects associated with inputs 
			ScalarObject so1 = (ScalarObject) pb.getVariable( input1.get_name(), input1.get_valueType() );
			ScalarObject so2 = null;
			if ( input2 != null )
				so2 = pb.getScalarVariable(input2.get_name(), input2.get_valueType() );
			
			// 2) Compute the result value & make an appropriate data object 
			SimpleOperator dop = (SimpleOperator) optr;
			
			if ( opcode.equalsIgnoreCase("print") ) {
				String buffer = "";
				if (input2 != null) {
					if (input2.get_valueType() != ValueType.STRING)
						throw new DMLRuntimeException("wrong value type in print");
					buffer = so2.getStringValue() + " ";
				}
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
			else if (opcode.equalsIgnoreCase("print2")) {
				System.out.println(so1.getStringValue());
			}
			else {
				/*
				 * Inputs for all builtins other than PRINT are treated as DOUBLE.
				 */
				double rval;
				if ( arity == 2 ) {
					rval = dop.fn.execute(so1.getDoubleValue(), so2.getDoubleValue());
				} 
				else {
					rval = dop.fn.execute(so1.getDoubleValue());
				}
				sores = (ScalarObject) new DoubleObject(rval);
			}
		}
		
		// 3) Put the result value into ProgramBlock
		pb.setVariable(output.get_name(), sores);
		return sores;
	}
}
