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

import java.util.ArrayList;

import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.data.OperationsOnMatrixValues;
import com.ibm.bi.dml.runtime.matrix.mapred.CachedValueMap;
import com.ibm.bi.dml.runtime.matrix.mapred.IndexedMatrixValue;
import com.ibm.bi.dml.runtime.matrix.operators.ScalarOperator;


public class ScalarInstruction extends UnaryMRInstructionBase 
{
	
	public ScalarInstruction(ScalarOperator op, byte in, byte out, String istr)
	{
		super(op, in, out);
		mrtype = MRINSTRUCTION_TYPE.ArithmeticBinary;
		instString = istr;
		
		//value dependent safe-safeness (trigger re-evaluation sparse-safe)
		op.setConstant(op.getConstant());
	}
	
	/**
	 * 
	 * @param str
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static Instruction parseInstruction ( String str )
		throws DMLRuntimeException 
	{	
		InstructionUtils.checkNumFields ( str, 3 );
		
		String[] parts = InstructionUtils.getInstructionParts ( str );
		String opcode = parts[0];
		boolean firstArgScalar = isFirstArgumentScalar(str);
		double cst = Double.parseDouble( firstArgScalar ? parts[1] : parts[2]);
		byte in = Byte.parseByte( firstArgScalar ? parts[2] : parts[1]);
		byte out = Byte.parseByte(parts[3]);
		
		ScalarOperator sop = InstructionUtils.parseScalarBinaryOperator(opcode, firstArgScalar, cst);
		return new ScalarInstruction(sop, in, out, str);
	}
	
	public void processInstruction(Class<? extends MatrixValue> valueClass, CachedValueMap cachedValues, 
			IndexedMatrixValue tempValue, IndexedMatrixValue zeroInput, int blockRowFactor, int blockColFactor)
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		ArrayList<IndexedMatrixValue> blkList = cachedValues.get(input);
		if( blkList != null )
			for( IndexedMatrixValue in : blkList )
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
	
	/**
	 * 
	 * @param inst
	 * @return
	 */
	private static boolean isFirstArgumentScalar(String inst)
	{
		//get first argument
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(inst);
		String arg1 = parts[1];
		
		//get data type of first argument
		String[] subparts = arg1.split(Lop.VALUETYPE_PREFIX);
		DataType dt = DataType.valueOf(subparts[1]);
		
		return (dt == DataType.SCALAR);
	}
}
