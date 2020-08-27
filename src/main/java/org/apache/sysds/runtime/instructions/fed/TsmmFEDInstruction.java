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

import org.apache.sysds.lops.MMTSJ.MMTSJType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse;
import org.apache.sysds.runtime.controlprogram.federated.FederationUtils;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest.RequestType;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

import java.util.concurrent.Future;

public class TsmmFEDInstruction extends BinaryFEDInstruction {
	private final MMTSJType _type;
	@SuppressWarnings("unused")
	private final int _numThreads;
	
	public TsmmFEDInstruction(CPOperand in, CPOperand out, MMTSJType type, int k, String opcode, String istr) {
		super(FEDType.Tsmm, null, in, null, out, opcode, istr);
		_type = type;
		_numThreads = k;
	}
	
	public static TsmmFEDInstruction parseInstruction(String str) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		if(!opcode.equalsIgnoreCase("tsmm"))
			throw new DMLRuntimeException("TsmmFedInstruction.parseInstruction():: Unknown opcode " + opcode);
		
		InstructionUtils.checkNumFields(parts, 4);
		CPOperand in = new CPOperand(parts[1]);
		CPOperand out = new CPOperand(parts[2]);
		MMTSJType type = MMTSJType.valueOf(parts[3]);
		int k = Integer.parseInt(parts[4]);
		return new TsmmFEDInstruction(in, out, type, k, opcode, str);
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) {
		MatrixObject mo1 = ec.getMatrixObject(input1);
		
		if(mo1.isFederated() && _type.isLeft()) { // left tsmm
			//construct commands: fed tsmm, retrieve results
			FederatedRequest fr1 = FederationUtils.callInstruction(instString, output,
				new CPOperand[]{input1}, new long[]{mo1.getFedMapping().getID()});
			FederatedRequest fr2 = new FederatedRequest(RequestType.GET_VAR, fr1.getID());
			FederatedRequest fr3 = mo1.getFedMapping().cleanup(getTID(), fr1.getID());
			
			//execute federated operations and aggregate
			Future<FederatedResponse>[] tmp = mo1.getFedMapping().execute(getTID(), fr1, fr2, fr3);
			MatrixBlock ret = FederationUtils.aggAdd(tmp);
			ec.setMatrixOutput(output.getName(), ret);
		}
		else { //other combinations
			throw new DMLRuntimeException("Federated Tsmm not supported with the "
				+ "following federated objects: "+mo1.isFederated()+" "+_fedType);
		}
	}
}
