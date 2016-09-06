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

import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.matrix.operators.Operator;
import org.apache.sysml.runtime.matrix.operators.SimpleOperator;
import org.apache.sysml.runtime.util.IndexRange;

public abstract class IndexingCPInstruction extends UnaryCPInstruction
{
	protected CPOperand rowLower, rowUpper, colLower, colUpper;
	
	public IndexingCPInstruction(Operator op, CPOperand in, CPOperand rl, CPOperand ru, CPOperand cl, CPOperand cu, CPOperand out, String opcode, String istr){
		super(op, in, out, opcode, istr);
		rowLower = rl;
		rowUpper = ru;
		colLower = cl;
		colUpper = cu;
	}
	
	public IndexingCPInstruction(Operator op, CPOperand lhsInput, CPOperand rhsInput, CPOperand rl, CPOperand ru, CPOperand cl, CPOperand cu, CPOperand out, String opcode, String istr){
		super(op, lhsInput, rhsInput, out, opcode, istr);
		rowLower = rl;
		rowUpper = ru;
		colLower = cl;
		colUpper = cu;
	}
	
	/**
	 * 
	 * @param ec
	 * @return
	 * @throws DMLRuntimeException
	 */
	protected IndexRange getIndexRange(ExecutionContext ec) throws DMLRuntimeException {
		return new IndexRange( //rl, ru, cl, ru
			(int)(ec.getScalarInput(rowLower.getName(), rowLower.getValueType(), rowLower.isLiteral()).getLongValue()-1),
			(int)(ec.getScalarInput(rowUpper.getName(), rowUpper.getValueType(), rowUpper.isLiteral()).getLongValue()-1),
			(int)(ec.getScalarInput(colLower.getName(), colLower.getValueType(), colLower.isLiteral()).getLongValue()-1),
			(int)(ec.getScalarInput(colUpper.getName(), colUpper.getValueType(), colUpper.isLiteral()).getLongValue()-1));		
	}
	
	/**
	 * 
	 * @param str
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static IndexingCPInstruction parseInstruction ( String str ) 
		throws DMLRuntimeException 
	{	
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		
		if ( opcode.equalsIgnoreCase("rangeReIndex") ) {
			if ( parts.length == 7 ) {
				// Example: rangeReIndex:mVar1:Var2:Var3:Var4:Var5:mVar6
				CPOperand in, rl, ru, cl, cu, out;
				in = new CPOperand();
				rl = new CPOperand();
				ru = new CPOperand();
				cl = new CPOperand();
				cu = new CPOperand();
				out = new CPOperand();
				in.split(parts[1]);
				rl.split(parts[2]);
				ru.split(parts[3]);
				cl.split(parts[4]);
				cu.split(parts[5]);
				out.split(parts[6]);
				if( in.getDataType()==DataType.MATRIX )
					return new MatrixIndexingCPInstruction(new SimpleOperator(null), in, rl, ru, cl, cu, out, opcode, str);
				else if (in.getDataType() == DataType.FRAME)
					return new FrameIndexingCPInstruction(new SimpleOperator(null), in, rl, ru, cl, cu, out, opcode, str);
				else 
					throw new DMLRuntimeException("Can index only on Frames or Matrices");
			}
			else {
				throw new DMLRuntimeException("Invalid number of operands in instruction: " + str);
			}
		} 
		else if ( opcode.equalsIgnoreCase("leftIndex")) {
			if ( parts.length == 8 ) {
				// Example: leftIndex:mVar1:mvar2:Var3:Var4:Var5:Var6:mVar7
				CPOperand lhsInput, rhsInput, rl, ru, cl, cu, out;
				lhsInput = new CPOperand();
				rhsInput = new CPOperand();
				rl = new CPOperand();
				ru = new CPOperand();
				cl = new CPOperand();
				cu = new CPOperand();
				out = new CPOperand();
				lhsInput.split(parts[1]);
				rhsInput.split(parts[2]);
				rl.split(parts[3]);
				ru.split(parts[4]);
				cl.split(parts[5]);
				cu.split(parts[6]);
				out.split(parts[7]);
				if( lhsInput.getDataType()==DataType.MATRIX )
					return new MatrixIndexingCPInstruction(new SimpleOperator(null), lhsInput, rhsInput, rl, ru, cl, cu, out, opcode, str);
				else if (lhsInput.getDataType() == DataType.FRAME)
					return new FrameIndexingCPInstruction(new SimpleOperator(null), lhsInput, rhsInput, rl, ru, cl, cu, out, opcode, str);
				else 
					throw new DMLRuntimeException("Can index only on Frames or Matrices");
			}
			else {
				throw new DMLRuntimeException("Invalid number of operands in instruction: " + str);
			}
		}
		else {
			throw new DMLRuntimeException("Unknown opcode while parsing a MatrixIndexingCPInstruction: " + str);
		}
	}
}
