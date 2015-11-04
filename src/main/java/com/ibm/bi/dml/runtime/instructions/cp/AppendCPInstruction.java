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
		RBIND,
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
		throws DMLRuntimeException 
	{
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		InstructionUtils.checkNumFields (parts, 5);
		
		String opcode = parts[0];
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand in2 = new CPOperand(parts[2]);
		CPOperand in3 = new CPOperand(parts[3]);
		CPOperand out = new CPOperand(parts[4]);
		boolean cbind = Boolean.parseBoolean(parts[5]);
		
		AppendType type = (in1.getDataType()!=DataType.MATRIX) ? AppendType.STRING :
						  cbind ? AppendType.CBIND : AppendType.RBIND;
		
		if(!opcode.equalsIgnoreCase("append"))
			throw new DMLRuntimeException("Unknown opcode while parsing a AppendCPInstruction: " + str);

		return new AppendCPInstruction(
				new ReorgOperator(OffsetColumnIndex.getOffsetColumnIndexFnObject(-1)), 
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
			if(matBlock1.getNumRows() != matBlock2.getNumRows()) {
				throw new DMLRuntimeException("Append-cbind is not possible for input matrices " 
											  + input1.getName() + " and " + input2.getName()
											  + " with different number of rows: "+matBlock1.getNumRows()+" vs "+matBlock2.getNumRows());
			}
				
			//execute append operations (append both inputs to initially empty output)
			MatrixBlock ret = matBlock1.appendOperations(matBlock2, new MatrixBlock(), true);
			
			//set output and release inputs 
			ec.setMatrixOutput(output.getName(), ret);
			ec.releaseMatrixInput(input1.getName());
			ec.releaseMatrixInput(input2.getName());
		}
		else if( _type == AppendType.RBIND )
		{
			//get inputs
			MatrixBlock matBlock1 = ec.getMatrixInput(input1.getName());
			MatrixBlock matBlock2 = ec.getMatrixInput(input2.getName());
			
			//check input dimensions
			if(matBlock1.getNumColumns() != matBlock2.getNumColumns()) {
				throw new DMLRuntimeException("Append-rbind is not possible for input matrices " 
											  + input1.getName() + " and " + input2.getName()
											  + " with different number of columns: "+matBlock1.getNumColumns()+" vs "+matBlock2.getNumColumns());
			}
			
			//execute append operations (append both inputs to initially empty output)
			MatrixBlock ret = matBlock1.appendOperations(matBlock2, new MatrixBlock(), false);
			
			//set output and release inputs 
			ec.setMatrixOutput(output.getName(), ret);
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
