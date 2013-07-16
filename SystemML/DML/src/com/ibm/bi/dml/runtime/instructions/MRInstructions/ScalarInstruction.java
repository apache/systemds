package com.ibm.bi.dml.runtime.instructions.MRInstructions;

import com.ibm.bi.dml.runtime.functionobjects.Builtin;
import com.ibm.bi.dml.runtime.functionobjects.Divide;
import com.ibm.bi.dml.runtime.functionobjects.EqualsReturnDouble;
import com.ibm.bi.dml.runtime.functionobjects.GreaterThanEqualsReturnDouble;
import com.ibm.bi.dml.runtime.functionobjects.GreaterThanReturnDouble;
import com.ibm.bi.dml.runtime.functionobjects.LessThanEqualsReturnDouble;
import com.ibm.bi.dml.runtime.functionobjects.LessThanReturnDouble;
import com.ibm.bi.dml.runtime.functionobjects.Minus;
import com.ibm.bi.dml.runtime.functionobjects.Multiply;
import com.ibm.bi.dml.runtime.functionobjects.NotEqualsReturnDouble;
import com.ibm.bi.dml.runtime.functionobjects.Plus;
import com.ibm.bi.dml.runtime.functionobjects.Power;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.io.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.io.OperationsOnMatrixValues;
import com.ibm.bi.dml.runtime.matrix.mapred.CachedValueMap;
import com.ibm.bi.dml.runtime.matrix.mapred.IndexedMatrixValue;
import com.ibm.bi.dml.runtime.matrix.operators.LeftScalarOperator;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.matrix.operators.RightScalarOperator;
import com.ibm.bi.dml.runtime.matrix.operators.ScalarOperator;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.DMLUnsupportedOperationException;


public class ScalarInstruction extends UnaryMRInstructionBase {

	public ScalarInstruction(Operator op, byte in, byte out, String istr)
	{
		super(op, in, out);
		mrtype = MRINSTRUCTION_TYPE.ArithmeticBinary;
		instString = istr;
	}
	
	public static Instruction parseInstruction ( String str ) throws DMLRuntimeException {
		
		InstructionUtils.checkNumFields ( str, 3 );
		
		String[] parts = InstructionUtils.getInstructionParts ( str );
		
		byte in, out;
		double cst;
		String opcode = parts[0];
		in = Byte.parseByte(parts[1]);
		cst = Double.parseDouble(parts[2]);
		out = Byte.parseByte(parts[3]);
		
		if ( opcode.equalsIgnoreCase("+") ) {
			return new ScalarInstruction(new RightScalarOperator(Plus.getPlusFnObject(), cst), in, out, str);
		} 
		else if ( opcode.equalsIgnoreCase("-") ) {
			return new ScalarInstruction(new RightScalarOperator(Minus.getMinusFnObject(), cst), in, out, str);
		} 
		else if ( opcode.equalsIgnoreCase("s-r") ) {
			return new ScalarInstruction(new LeftScalarOperator(Minus.getMinusFnObject(), cst), in, out, str);
		} 
		else if ( opcode.equalsIgnoreCase("*") ) {
			return new ScalarInstruction(new RightScalarOperator(Multiply.getMultiplyFnObject(), cst), in, out, str);
		} 
		else if ( opcode.equalsIgnoreCase("/") ) {
			return new ScalarInstruction(new RightScalarOperator(Divide.getDivideFnObject(), cst), in, out, str);
		} 
		else if ( opcode.equalsIgnoreCase("so") ) {
			return new ScalarInstruction(new LeftScalarOperator(Divide.getDivideFnObject(), cst), in, out, str);
		} 
		else if ( opcode.equalsIgnoreCase("^") ) {
			return new ScalarInstruction(new RightScalarOperator(Power.getPowerFnObject(), cst), in, out, str);
		}
		else if ( opcode.equalsIgnoreCase("max") ) {
			return new ScalarInstruction(new RightScalarOperator(Builtin.getBuiltinFnObject("max"), cst), in, out, str);
		}
		else if ( opcode.equalsIgnoreCase("min") ) {
			return new ScalarInstruction(new RightScalarOperator(Builtin.getBuiltinFnObject("min"), cst), in, out, str);
		}
		else if ( opcode.equalsIgnoreCase(">") ) {
			return new ScalarInstruction(new RightScalarOperator(GreaterThanReturnDouble.getGreaterThanReturnDoubleFnObject(), cst), in, out, str);
		}
		else if ( opcode.equalsIgnoreCase(">=") ) {
			return new ScalarInstruction(new RightScalarOperator(GreaterThanEqualsReturnDouble.getGreaterThanEqualsReturnDoubleFnObject(), cst), in, out, str);
		}
		else if ( opcode.equalsIgnoreCase("<") ) {
			return new ScalarInstruction(new RightScalarOperator(LessThanReturnDouble.getLessThanReturnDoubleFnObject(), cst), in, out, str);
		}
		else if ( opcode.equalsIgnoreCase("<=") ) {
			return new ScalarInstruction(new RightScalarOperator(LessThanEqualsReturnDouble.getLessThanEqualsReturnDoubleFnObject(), cst), in, out, str);
		}
		else if ( opcode.equalsIgnoreCase("==") ) {
			return new ScalarInstruction(new RightScalarOperator(EqualsReturnDouble.getEqualsReturnDoubleFnObject(), cst), in, out, str);
		}
		else if ( opcode.equalsIgnoreCase("!=") ) {
			return new ScalarInstruction(new RightScalarOperator(NotEqualsReturnDouble.getNotEqualsReturnDoubleFnObject(), cst), in, out, str);
		}
		
		return null;
	}
	
	public void processInstruction(Class<? extends MatrixValue> valueClass, CachedValueMap cachedValues, 
			IndexedMatrixValue tempValue, IndexedMatrixValue zeroInput, int blockRowFactor, int blockColFactor)
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		for(IndexedMatrixValue in: cachedValues.get(input))
		{
			if(in==null)
				continue;
		
			//allocate space for the output value
			IndexedMatrixValue out;
			if(input==output)
				out=tempValue;
			else
				out=cachedValues.holdPlace(output, valueClass);
			
			//process instruction
			out.getIndexes().setIndexes(in.getIndexes());
			OperationsOnMatrixValues.performScalarIgnoreIndexes(in.getValue(), out.getValue(), ((ScalarOperator)this.optr));
			
			//put the output value in the cache
			if(out==tempValue)
				cachedValues.add(output, out);
		}
	}
}
