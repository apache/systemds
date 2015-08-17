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
import com.ibm.bi.dml.runtime.functionobjects.CM;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.operators.CMOperator;
import com.ibm.bi.dml.runtime.matrix.operators.CMOperator.AggregateOperationTypes;

public class CentralMomentCPInstruction extends AggregateUnaryCPInstruction
{
	
	
	public CentralMomentCPInstruction(CMOperator cm, CPOperand in1, CPOperand in2, 
			CPOperand in3, CPOperand out, String opcode, String str) 
	{
		super(cm, in1, in2, in3, out, opcode, str);
	}

	public static Instruction parseInstruction(String str)
		throws DMLRuntimeException 
	{
		CPOperand in1 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand in2 = null; 
		CPOperand in3 = null; 
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		
		String opcode = InstructionUtils.getOpCode(str); 
		
		//check supported opcode
		if( !opcode.equalsIgnoreCase("cm") ) {
			throw new DMLRuntimeException("Unsupported opcode "+opcode);
		}
			
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		if ( parts.length == 4 ) {
			// Example: CP.cm.mVar0.Var1.mVar2; (without weights)
			in2 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
			parseUnaryInstruction(str, in1, in2, out);
		}
		else if ( parts.length == 5) {
			// CP.cm.mVar0.mVar1.Var2.mVar3; (with weights)
			in2 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
			in3 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
			parseUnaryInstruction(str, in1, in2, in3, out);
		}
	
		/* 
		 * Exact order of the central moment MAY NOT be known at compilation time.
		 * We first try to parse the second argument as an integer, and if we fail, 
		 * we simply pass -1 so that getCMAggOpType() picks up AggregateOperationTypes.INVALID.
		 * It must be updated at run time in processInstruction() method.
		 */
		
		int cmOrder;
		try {
			if ( in3 == null ) {
				cmOrder = Integer.parseInt(in2.getName());
			}
			else {
				cmOrder = Integer.parseInt(in3.getName());
			}
		} catch(NumberFormatException e) {
			cmOrder = -1; // unknown at compilation time
		}
		
		AggregateOperationTypes opType = CMOperator.getCMAggOpType(cmOrder);
		CMOperator cm = new CMOperator(CM.getCMFnObject(opType), opType);
		return new CentralMomentCPInstruction(cm, in1, in2, in3, out, opcode, str);
	}
	
	@Override
	public void processInstruction( ExecutionContext ec )
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		String output_name = output.getName();

		/*
		 * The "order" of the central moment in the instruction can 
		 * be set to INVALID when the exact value is unknown at 
		 * compilation time. We first need to determine the exact 
		 * order and update the CMOperator, if needed.
		 */
		
		MatrixBlock matBlock = ec.getMatrixInput(input1.getName());

		CPOperand scalarInput = (input3==null ? input2 : input3);
		ScalarObject order = ec.getScalarInput(scalarInput.getName(), scalarInput.getValueType(), scalarInput.isLiteral()); 
		
		CMOperator cm_op = ((CMOperator)_optr); 
		if ( cm_op.getAggOpType() == AggregateOperationTypes.INVALID ) {
			((CMOperator)_optr).setCMAggOp((int)order.getLongValue());
		}
		
		CM_COV_Object cmobj = null; 
		if (input3 == null ) {
			cmobj = matBlock.cmOperations(cm_op);
		}
		else {
			MatrixBlock wtBlock = ec.getMatrixInput(input2.getName());
			cmobj = matBlock.cmOperations(cm_op, wtBlock);
			ec.releaseMatrixInput(input2.getName());
		}
		
		ec.releaseMatrixInput(input1.getName());
		
		double val = cmobj.getRequiredResult(_optr);
		DoubleObject ret = new DoubleObject(output_name, val);
		ec.setScalarOutput(output_name, ret);
	}
}
