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

package org.apache.sysds.runtime.instructions.fed;

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.LibFederatedAppend;
import org.apache.sysds.runtime.functionobjects.OffsetColumnIndex;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.matrix.operators.ReorgOperator;

public class AppendFEDInstruction extends BinaryFEDInstruction {
	public enum FEDAppendType {
		CBIND, RBIND;
		public boolean isCBind() {
			return this == CBIND;
		}
	}
	
	protected final FEDAppendType _type;
	
	protected AppendFEDInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out, FEDAppendType type,
			String opcode, String istr) {
		super(FEDType.Append, op, in1, in2, out, opcode, istr);
		_type = type;
	}
	
	public static AppendFEDInstruction parseInstruction(String str) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		InstructionUtils.checkNumFields(parts, 5, 4);
		
		String opcode = parts[0];
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand in2 = new CPOperand(parts[2]);
		CPOperand out = new CPOperand(parts[parts.length - 2]);
		boolean cbind = Boolean.parseBoolean(parts[parts.length - 1]);
		
		FEDAppendType type = cbind ? FEDAppendType.CBIND : FEDAppendType.RBIND;
		
		if (!opcode.equalsIgnoreCase("append") && !opcode.equalsIgnoreCase("remove") 
			&& !opcode.equalsIgnoreCase("galignedappend"))
			throw new DMLRuntimeException("Unknown opcode while parsing a AppendCPInstruction: " + str);
		
		Operator op = new ReorgOperator(OffsetColumnIndex.getOffsetColumnIndexFnObject(-1));
		return new AppendFEDInstruction(op, in1, in2, out, type, opcode, str);
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) {
		//get inputs
		MatrixObject matObject1 = ec.getMatrixObject(input1.getName());
		MatrixObject matObject2 = ec.getMatrixObject(input2.getName());
		//check input dimensions
		if (_type == FEDAppendType.CBIND && matObject1.getNumRows() != matObject2.getNumRows()) {
			throw new DMLRuntimeException(
				"Append-cbind is not possible for federated input matrices " + input1.getName() + " and "
				+ input2.getName() + " with different number of rows: " + matObject1.getNumRows() + " vs "
				+ matObject2.getNumRows());
		}
		else if (_type == FEDAppendType.RBIND && matObject1.getNumColumns() != matObject2.getNumColumns()) {
			throw new DMLRuntimeException(
				"Append-rbind is not possible for federated input matrices " + input1.getName() + " and "
				+ input2.getName() + " with different number of columns: " + matObject1.getNumColumns()
				+ " vs " + matObject2.getNumColumns());
		}
		// append MatrixObjects
		LibFederatedAppend.federateAppend(matObject1, matObject2,
			ec.getMatrixObject(output.getName()), _type.isCBind());
	}
}
