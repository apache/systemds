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

package com.ibm.bi.dml.runtime.instructions.cp;

import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.functionobjects.OffsetColumnIndex;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.matrix.operators.ReorgOperator;


public class AppendCPInstruction extends BinaryCPInstruction
{
	
	public enum AppendType{
		CBIND,
		STRING,
	}

	//type (matrix cbind / scalar string concatenation)
	private AppendType _type;
	
	public AppendCPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out, AppendType type, String opcode, String istr)
	{
		super(op, in1, in2, out, opcode, istr);
		_cptype = CPINSTRUCTION_TYPE.Append;
		
		_type = type;
	}
	
	public static Instruction parseInstruction ( String str ) 
		throws DMLRuntimeException {
		CPOperand in1 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand in2 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand in3 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		
		//4 parts to the instruction besides opcode and execlocation
		//two input args, one output arg and offset = 4
		InstructionUtils.checkNumFields ( str, 4 );
		
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		in1.split(parts[1]);
		in2.split(parts[2]);
		in3.split(parts[3]);
		out.split(parts[4]);
		//String offset_str = parts[4];
		 
		AppendType type = (in1.getDataType()==DataType.MATRIX) ? AppendType.CBIND : AppendType.STRING;
		
		
		if(!opcode.equalsIgnoreCase("append"))
			throw new DMLRuntimeException("Unknown opcode while parsing a AppendCPInstruction: " + str);
		else
			return new AppendCPInstruction(new ReorgOperator(OffsetColumnIndex.getOffsetColumnIndexFnObject(-1)), 
										   in1, in2, in3, out, type, opcode, str);
	}
	
	@Override
	public void processInstruction(ExecutionContext ec)
		throws DMLUnsupportedOperationException, DMLRuntimeException 
	{
		if( _type == AppendType.CBIND )
		{
			//get inputs
			MatrixBlock matBlock1 = ec.getMatrixInput(input1.getName());
			MatrixBlock matBlock2 = ec.getMatrixInput(input2.getName());
			
			//check input dimensions
			if(matBlock1.getNumRows() != matBlock2.getNumRows())
				throw new DMLRuntimeException("Append is not possible for input matrices " 
											  + input1.getName() + " and " + input2.getName()
											  + "with unequal number of rows");
			
			//execute append operations (append both inputs to initially empty output)
			MatrixBlock ret = matBlock1.appendOperations(matBlock2, new MatrixBlock());
			
			//set output
			ec.setMatrixOutput(output.getName(), ret);
			
			//release inputs 
			ec.releaseMatrixInput(input1.getName());
			ec.releaseMatrixInput(input2.getName());
		}
		else //STRING
		{
			//get input strings (vars or literals)
			ScalarObject so1 = ec.getScalarInput( input1.getName(), input1.getValueType(), input1.isLiteral() );
			ScalarObject so2 = ec.getScalarInput( input2.getName(), input2.getValueType(), input2.isLiteral() );
			
			//pre-checks
			String val1 = so1.getStringValue();
			String val2 = so2.getStringValue();
			StringObject.checkMaxStringLength( val1.length()+val2.length() );
			
			//core execution
			String outString = val1 + "\n" + val2;			
			ScalarObject sores = new StringObject(outString);
			
			//set output
			ec.setScalarOutput(output.getName(), sores);
		}		
	}
}
