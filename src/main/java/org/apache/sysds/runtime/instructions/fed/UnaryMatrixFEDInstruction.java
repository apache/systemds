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

import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest;
import org.apache.sysds.runtime.controlprogram.federated.FederationUtils;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.ValueFunction;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.matrix.data.LibCommonsMath;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;

public class UnaryMatrixFEDInstruction extends UnaryFEDInstruction {
	protected UnaryMatrixFEDInstruction(Operator op, CPOperand in, CPOperand out, String opcode, String instr) {
		super(FEDType.Unary, op, in, out, opcode, instr);
	}
	
	public static boolean isValidOpcode(String opcode) {
		return !LibCommonsMath.isSupportedUnaryOperation(opcode)
			&& !opcode.startsWith("ucum"); //ucumk+ ucum* ucumk+* ucummin ucummax
	}

	public static UnaryMatrixFEDInstruction parseInstruction(String str) {
		CPOperand in = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);

		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode;
		opcode = parts[0];
		if( opcode.equalsIgnoreCase("exp") && parts.length == 5) {
			in.split(parts[1]);
			out.split(parts[2]);
			ValueFunction func = Builtin.getBuiltinFnObject(opcode);
			return new UnaryMatrixFEDInstruction(new UnaryOperator(func,
				Integer.parseInt(parts[3]),Boolean.parseBoolean(parts[4])), in, out, opcode, str);
		}
		opcode = parseUnaryInstruction(str, in, out);
		return new UnaryMatrixFEDInstruction(InstructionUtils.parseUnaryOperator(opcode), in, out, opcode, str);
	}
	
	@Override 
	public void processInstruction(ExecutionContext ec) {
		MatrixObject mo1 = ec.getMatrixObject(input1);
		
		//federated execution on arbitrary row/column partitions
		//(only assumption for sparse-unsafe: fed mapping covers entire matrix)
		FederatedRequest fr1 = FederationUtils.callInstruction(instString, output,
			new CPOperand[]{input1}, new long[]{mo1.getFedMapping().getID()});
		mo1.getFedMapping().execute(getTID(), true, fr1);
		
		//set characteristics and fed mapp
		MatrixObject out = ec.getMatrixObject(output);
		out.getDataCharacteristics().set(mo1.getNumRows(), mo1.getNumColumns(), (int)mo1.getBlocksize());
		out.setFedMapping(mo1.getFedMapping().copyWithNewID(fr1.getID()));
	}
}
