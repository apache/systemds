package dml.runtime.instructions.MRInstructions;

import dml.runtime.functionobjects.Builtin;
import dml.runtime.functionobjects.Divide;
import dml.runtime.functionobjects.EqualsReturnDouble;
import dml.runtime.functionobjects.GreaterThanEqualsReturnDouble;
import dml.runtime.functionobjects.GreaterThanReturnDouble;
import dml.runtime.functionobjects.LessThanEqualsReturnDouble;
import dml.runtime.functionobjects.LessThanReturnDouble;
import dml.runtime.functionobjects.Minus;
import dml.runtime.functionobjects.Multiply;
import dml.runtime.functionobjects.NotEqualsReturnDouble;
import dml.runtime.functionobjects.Plus;
import dml.runtime.instructions.Instruction;
import dml.runtime.instructions.InstructionUtils;
import dml.runtime.matrix.io.MatrixIndexes;
import dml.runtime.matrix.io.MatrixValue;
import dml.runtime.matrix.io.OperationsOnMatrixValues;
import dml.runtime.matrix.mapred.CachedValueMap;
import dml.runtime.matrix.mapred.IndexedMatrixValue;
import dml.runtime.matrix.operators.BinaryOperator;
import dml.runtime.matrix.operators.Operator;
import dml.utils.DMLRuntimeException;
import dml.utils.DMLUnsupportedOperationException;

public class BinaryInstruction extends BinaryMRInstructionBase {

	public BinaryInstruction(Operator op, byte in1, byte in2, byte out, String istr)
	{
		super(op, in1, in2, out);
		mrtype = MRINSTRUCTION_TYPE.Binary;
		instString = istr;
	}
	
	public static Instruction parseInstruction ( String str ) throws DMLRuntimeException {
		
		InstructionUtils.checkNumFields ( str, 3 );
		
		String[] parts = InstructionUtils.getInstructionParts ( str );
		
		byte in1, in2, out;
		String opcode = parts[0];
		in1 = Byte.parseByte(parts[1]);
		in2 = Byte.parseByte(parts[2]);
		out = Byte.parseByte(parts[3]);
		
		if ( opcode.equalsIgnoreCase("+") ) {
			return new BinaryInstruction(new BinaryOperator(Plus.getPlusFnObject()), in1, in2, out, str);
		} 
		else if ( opcode.equalsIgnoreCase("-") ) {
			return new BinaryInstruction(new BinaryOperator(Minus.getMinusFnObject()), in1, in2, out, str);
		}
		else if ( opcode.equalsIgnoreCase("*") ) {
			return new BinaryInstruction(new BinaryOperator(Multiply.getMultiplyFnObject()), in1, in2, out, str);
		}
		else if ( opcode.equalsIgnoreCase("/") ) {
			return new BinaryInstruction(new BinaryOperator(Divide.getDivideFnObject()), in1, in2, out, str);
		}
		else if ( opcode.equalsIgnoreCase("max") ) {
			return new BinaryInstruction(new BinaryOperator(Builtin.getBuiltinFnObject("max")), in1, in2, out, str);
		}
		else if ( opcode.equalsIgnoreCase(">") ) {
			return new BinaryInstruction(new BinaryOperator(GreaterThanReturnDouble.getGreaterThanReturnDoubleFnObject()), in1, in2, out, str);
		}
		else if ( opcode.equalsIgnoreCase(">=") ) {
			return new BinaryInstruction(new BinaryOperator(GreaterThanEqualsReturnDouble.getGreaterThanEqualsReturnDoubleFnObject()), in1, in2, out, str);
		}
		else if ( opcode.equalsIgnoreCase("<") ) {
			return new BinaryInstruction(new BinaryOperator(LessThanReturnDouble.getLessThanReturnDoubleFnObject()), in1, in2, out, str);
		}
		else if ( opcode.equalsIgnoreCase("<=") ) {
			return new BinaryInstruction(new BinaryOperator(LessThanEqualsReturnDouble.getLessThanEqualsReturnDoubleFnObject()), in1, in2, out, str);
		}
		else if ( opcode.equalsIgnoreCase("==") ) {
			return new BinaryInstruction(new BinaryOperator(EqualsReturnDouble.getEqualsReturnDoubleFnObject()), in1, in2, out, str);
		}
		else if ( opcode.equalsIgnoreCase("!=") ) {
			return new BinaryInstruction(new BinaryOperator(NotEqualsReturnDouble.getNotEqualsReturnDoubleFnObject()), in1, in2, out, str);
		}
		return null;
	}
	
	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass,
			CachedValueMap cachedValues, IndexedMatrixValue tempValue, IndexedMatrixValue zeroInput,
			int blockRowFactor, int blockColFactor)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		
		IndexedMatrixValue in1=cachedValues.get(input1);
		IndexedMatrixValue in2=cachedValues.get(input2);
		if(in1==null && in2==null)
			return;
		
		//allocate space for the output value
		//try to avoid coping as much as possible
		IndexedMatrixValue out;
		if( (output!=input1 && output!=input2)
			|| (output==input1 && in1==null)
			|| (output==input2 && in2==null) )
			out=cachedValues.holdPlace(output, valueClass);
		else
			out=tempValue;
		
		//if one of the inputs is null, then it is a all zero block
		MatrixIndexes finalIndexes=null;
		if(in1==null)
		{
			in1=zeroInput;
			in1.getValue().reset(in2.getValue().getNumRows(), 
					in2.getValue().getNumColumns());
			finalIndexes=in2.getIndexes();
		}else
			finalIndexes=in1.getIndexes();
		
		if(in2==null)
		{
			in2=zeroInput;
			in2.getValue().reset(in1.getValue().getNumRows(), 
					in1.getValue().getNumColumns());
		}
		
		//process instruction
		out.getIndexes().setIndexes(finalIndexes);
		OperationsOnMatrixValues.performBinaryIgnoreIndexes(in1.getValue(), 
				in2.getValue(), out.getValue(), ((BinaryOperator)optr));
		
		//put the output value in the cache
		if(out==tempValue)
			cachedValues.set(output, out);
		
	}

}
