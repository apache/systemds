/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysml.runtime.instructions.cp;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.DMLUnsupportedOperationException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.functionobjects.Builtin;
import org.apache.sysml.runtime.instructions.Instruction;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysml.runtime.matrix.operators.Operator;
import org.apache.sysml.runtime.matrix.operators.SimpleOperator;

public class AggregateUnaryCPInstruction extends UnaryCPInstruction
{
	
	public AggregateUnaryCPInstruction(Operator op, CPOperand in, CPOperand out, String opcode, String istr){
		this(op, in, null, null, out, opcode, istr);
	}
	
	public AggregateUnaryCPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out, String opcode, String istr){
		this(op, in1, in2, null, out, opcode, istr);
	}
	
	public AggregateUnaryCPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out, String opcode, String istr){
		super(op, in1, in2, in3, out, opcode, istr);
		_cptype = CPINSTRUCTION_TYPE.AggregateUnary;		
	}
	
	public static Instruction parseInstruction(String str)
		throws DMLRuntimeException 
	{
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];		
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand out = new CPOperand(parts[2]);
		
		if(opcode.equalsIgnoreCase("nrow") || opcode.equalsIgnoreCase("ncol") || opcode.equalsIgnoreCase("length")){
			return new AggregateUnaryCPInstruction(new SimpleOperator(Builtin.getBuiltinFnObject(opcode)),
												   in1,  out, opcode, str);
		}
		else //DEFAULT BEHAVIOR
		{
			AggregateUnaryOperator aggun = InstructionUtils.parseBasicAggregateUnaryOperator(opcode);
			aggun.setNumThreads( Integer.parseInt(parts[3]) );
			return new AggregateUnaryCPInstruction(aggun, in1, out, opcode, str);				
		}
	}
	
	@Override
	public void processInstruction( ExecutionContext ec )
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		String output_name = output.getName();
		String opcode = getOpcode();
		
		if( opcode.equalsIgnoreCase("nrow") || opcode.equalsIgnoreCase("ncol") || opcode.equalsIgnoreCase("length")  )
		{
			//check existence of input variable
			if( !ec.getVariables().keySet().contains(input1.getName()) ){
				throw new DMLRuntimeException("Variable '"+input1.getName()+"' does not exist.");
			}
			
			//get meta data information
			MatrixCharacteristics mc = ec.getMatrixCharacteristics(input1.getName());
			long rval = -1;
			if(opcode.equalsIgnoreCase("nrow"))
				rval = mc.getRows();
			else if(opcode.equalsIgnoreCase("ncol"))
				rval = mc.getCols();
			else if(opcode.equalsIgnoreCase("length"))
				rval = mc.getRows() * mc.getCols();

			//check for valid output, and acquire read if necessary
			//(Use case: In case of forced exec type singlenode, there are no reblocks. For csv
			//we however, support unspecified input sizes, which requires a read to obtain the
			//required meta data)
			//Note: check on matrix characteristics to cover incorrect length (-1*-1 -> 1)
			if( !mc.dimsKnown() ) //invalid nrow/ncol/length
			{
				if( DMLScript.rtplatform == RUNTIME_PLATFORM.SINGLE_NODE )
				{
					//read the input data and explicitly refresh input data
					MatrixObject mo = (MatrixObject)ec.getVariable(input1.getName());
					mo.acquireRead();
					mo.refreshMetaData();
					mo.release();
					
					//update meta data information
					mc = ec.getMatrixCharacteristics(input1.getName());
					if(opcode.equalsIgnoreCase("nrow"))
						rval = mc.getRows();
					else if(opcode.equalsIgnoreCase("ncol"))
						rval = mc.getCols();
					else if(opcode.equalsIgnoreCase("length"))
						rval = mc.getRows() * mc.getCols();
				}
				else {
					throw new DMLRuntimeException("Invalid meta data returned by '"+opcode+"': "+rval);
				}
			}
			
			//create and set output scalar
			ScalarObject ret = null;
			switch( output.getValueType() ) {
				case INT:	  ret = new IntObject(output_name, rval); break;
				case DOUBLE:  ret = new DoubleObject(output_name, rval); break;
				case STRING:  ret = new StringObject(output_name, String.valueOf(rval)); break;
				
				default: 
					throw new DMLRuntimeException("Invalid output value type: "+output.getValueType());
			}
			ec.setScalarOutput(output_name, ret);
			return;
		}
		else 
		{
			/* Default behavior for AggregateUnary Instruction */
			MatrixBlock matBlock = ec.getMatrixInput(input1.getName());		
			AggregateUnaryOperator au_op = (AggregateUnaryOperator) _optr;
			
			MatrixBlock resultBlock = (MatrixBlock) matBlock.aggregateUnaryOperations(au_op, new MatrixBlock(), matBlock.getNumRows(), matBlock.getNumColumns(), new MatrixIndexes(1, 1), true);
			
			ec.releaseMatrixInput(input1.getName());
			
			if(output.getDataType() == DataType.SCALAR){
				DoubleObject ret = new DoubleObject(output_name, resultBlock.getValue(0, 0));
				ec.setScalarOutput(output_name, ret);
			} else{
				// since the computed value is a scalar, allocate a "temp" output matrix
				ec.setMatrixOutput(output_name, resultBlock);
			}
		}
	}

}
