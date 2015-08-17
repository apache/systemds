/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package com.ibm.bi.dml.runtime.instructions.mr;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.functionobjects.Builtin;
import com.ibm.bi.dml.runtime.functionobjects.Divide;
import com.ibm.bi.dml.runtime.functionobjects.Equals;
import com.ibm.bi.dml.runtime.functionobjects.GreaterThan;
import com.ibm.bi.dml.runtime.functionobjects.GreaterThanEquals;
import com.ibm.bi.dml.runtime.functionobjects.IntegerDivide;
import com.ibm.bi.dml.runtime.functionobjects.LessThan;
import com.ibm.bi.dml.runtime.functionobjects.LessThanEquals;
import com.ibm.bi.dml.runtime.functionobjects.Minus;
import com.ibm.bi.dml.runtime.functionobjects.Modulus;
import com.ibm.bi.dml.runtime.functionobjects.Multiply;
import com.ibm.bi.dml.runtime.functionobjects.NotEquals;
import com.ibm.bi.dml.runtime.functionobjects.Plus;
import com.ibm.bi.dml.runtime.functionobjects.Power;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.data.OperationsOnMatrixValues;
import com.ibm.bi.dml.runtime.matrix.mapred.CachedValueMap;
import com.ibm.bi.dml.runtime.matrix.mapred.IndexedMatrixValue;
import com.ibm.bi.dml.runtime.matrix.operators.BinaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;


public class BinaryInstruction extends BinaryMRInstructionBase 
{
	
	public BinaryInstruction(Operator op, byte in1, byte in2, byte out, String istr)
	{
		super(op, in1, in2, out);
		mrtype = MRINSTRUCTION_TYPE.ArithmeticBinary;
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
		
		BinaryOperator bop = parseBinaryOperator(opcode);
		if( bop != null )
			return new BinaryInstruction(bop, in1, in2, out, str);
		else
			return null;
	}

	/**
	 * 
	 * @param opcode
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static BinaryOperator parseBinaryOperator( String opcode ) 
		throws DMLRuntimeException 
	{
		if ( opcode.equalsIgnoreCase("+") ) {
			return new BinaryOperator(Plus.getPlusFnObject());
		} 
		else if ( opcode.equalsIgnoreCase("-") ) {
			return new BinaryOperator(Minus.getMinusFnObject());
		}
		else if ( opcode.equalsIgnoreCase("*") ) {
			return new BinaryOperator(Multiply.getMultiplyFnObject());
		}
		else if ( opcode.equalsIgnoreCase("/") ) {
			return new BinaryOperator(Divide.getDivideFnObject());
		}
		else if ( opcode.equalsIgnoreCase("^") ) {
			return new BinaryOperator(Power.getPowerFnObject());
		}
		else if ( opcode.equalsIgnoreCase("%%") ) {
			return new BinaryOperator(Modulus.getModulusFnObject());
		}
		else if ( opcode.equalsIgnoreCase("%/%") ) {
			return new BinaryOperator(IntegerDivide.getIntegerDivideFnObject());
		}
		else if ( opcode.equalsIgnoreCase("min") ) {
			return new BinaryOperator(Builtin.getBuiltinFnObject("min"));
		}
		else if ( opcode.equalsIgnoreCase("max") ) {
			return new BinaryOperator(Builtin.getBuiltinFnObject("max"));
		}
		else if ( opcode.equalsIgnoreCase(">") ) {
			return new BinaryOperator(GreaterThan.getGreaterThanFnObject());
		}
		else if ( opcode.equalsIgnoreCase(">=") ) {
			return new BinaryOperator(GreaterThanEquals.getGreaterThanEqualsFnObject());
		}
		else if ( opcode.equalsIgnoreCase("<") ) {
			return new BinaryOperator(LessThan.getLessThanFnObject());
		}
		else if ( opcode.equalsIgnoreCase("<=") ) {
			return new BinaryOperator(LessThanEquals.getLessThanEqualsFnObject());
		}
		else if ( opcode.equalsIgnoreCase("==") ) {
			return new BinaryOperator(Equals.getEqualsFnObject());
		}
		else if ( opcode.equalsIgnoreCase("!=") ) {
			return new BinaryOperator(NotEquals.getNotEqualsFnObject());
		}
		return null;
	}
	
	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass,
			CachedValueMap cachedValues, IndexedMatrixValue tempValue, IndexedMatrixValue zeroInput,
			int blockRowFactor, int blockColFactor)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		
		IndexedMatrixValue in1=cachedValues.getFirst(input1);
		IndexedMatrixValue in2=cachedValues.getFirst(input2);
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
			cachedValues.add(output, out);
		
	}

}
