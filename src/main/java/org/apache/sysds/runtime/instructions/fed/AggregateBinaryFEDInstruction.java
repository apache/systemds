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
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse;
import org.apache.sysds.runtime.controlprogram.federated.FederationUtils;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest.RequestType;
import org.apache.sysds.runtime.controlprogram.federated.FederationMap.FType;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.Operator;

import java.util.concurrent.Future;

public class AggregateBinaryFEDInstruction extends BinaryFEDInstruction {
	
	public AggregateBinaryFEDInstruction(Operator op, CPOperand in1,
		CPOperand in2, CPOperand out, String opcode, String istr) {
		super(FEDType.AggregateBinary, op, in1, in2, out, opcode, istr);
	}
	
	public static AggregateBinaryFEDInstruction parseInstruction(String str) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		if(!opcode.equalsIgnoreCase("ba+*"))
			throw new DMLRuntimeException("AggregateBinaryInstruction.parseInstruction():: Unknown opcode " + opcode);
		
		InstructionUtils.checkNumFields(parts, 4);
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand in2 = new CPOperand(parts[2]);
		CPOperand out = new CPOperand(parts[3]);
		int k = Integer.parseInt(parts[4]);
		return new AggregateBinaryFEDInstruction(
			InstructionUtils.getMatMultOperator(k), in1, in2, out, opcode, str);
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) {
		MatrixObject mo1 = ec.getMatrixObject(input1);
		MatrixObject mo2 = ec.getMatrixObject(input2);
		
		//#1 federated matrix-vector multiplication
		if(mo1.isFederated(FType.COL) && mo2.isFederated(FType.ROW)
			&& mo1.getFedMapping().isAligned(mo2.getFedMapping(), true) ) {
			FederatedRequest fr1 = FederationUtils.callInstruction(instString, output,
				new CPOperand[]{input1, input2},
				new long[]{mo1.getFedMapping().getID(), mo2.getFedMapping().getID()});
			FederatedRequest fr2 = new FederatedRequest(RequestType.GET_VAR, fr1.getID());
			FederatedRequest fr3 = mo2.getFedMapping().cleanup(getTID(), fr1.getID(), fr2.getID());
			//execute federated operations and aggregate
			Future<FederatedResponse>[] tmp = mo1.getFedMapping().execute(getTID(), fr1, fr2, fr3);
			MatrixBlock ret = FederationUtils.aggAdd(tmp);
			ec.setMatrixOutput(output.getName(), ret);
		}
		else if(mo1.isFederated(FType.ROW)) { // MV + MM
			//construct commands: broadcast rhs, fed mv, retrieve results
			FederatedRequest fr1 = mo1.getFedMapping().broadcast(mo2);
			FederatedRequest fr2 = FederationUtils.callInstruction(instString, output,
				new CPOperand[]{input1, input2}, new long[]{mo1.getFedMapping().getID(), fr1.getID()});
			if( mo2.getNumColumns() == 1 ) { //MV
				FederatedRequest fr3 = new FederatedRequest(RequestType.GET_VAR, fr2.getID());
				FederatedRequest fr4 = mo1.getFedMapping().cleanup(getTID(), fr1.getID(), fr2.getID());
				//execute federated operations and aggregate
				Future<FederatedResponse>[] tmp = mo1.getFedMapping().execute(getTID(), fr1, fr2, fr3, fr4);
				MatrixBlock ret = FederationUtils.rbind(tmp);
				ec.setMatrixOutput(output.getName(), ret);
			}
			else { //MM
				//execute federated operations and aggregate
				FederatedRequest fr3 = mo1.getFedMapping().cleanup(getTID(), fr1.getID());
				mo1.getFedMapping().execute(getTID(), true, fr1, fr2, fr3);
				MatrixObject out = ec.getMatrixObject(output);
				out.getDataCharacteristics().set(mo1.getNumRows(), mo2.getNumColumns(), (int)mo1.getBlocksize());
				out.setFedMapping(mo1.getFedMapping().copyWithNewID(fr2.getID(), mo2.getNumColumns()));
				out.getFedMapping().setType(FType.ROW);
			}
		}
		//#2 vector - federated matrix multiplication
		else if (mo2.isFederated(FType.ROW)) {// VM + MM
			//construct commands: broadcast rhs, fed mv, retrieve results
			FederatedRequest[] fr1 = mo2.getFedMapping().broadcastSliced(mo1, true);
			FederatedRequest fr2 = FederationUtils.callInstruction(instString, output,
				new CPOperand[]{input1, input2}, new long[]{fr1[0].getID(), mo2.getFedMapping().getID()});
			FederatedRequest fr3 = new FederatedRequest(RequestType.GET_VAR, fr2.getID());
			FederatedRequest fr4 = mo2.getFedMapping().cleanup(getTID(), fr1[0].getID(), fr2.getID());
			//execute federated operations and aggregate
			Future<FederatedResponse>[] tmp = mo2.getFedMapping().execute(getTID(), fr1, fr2, fr3, fr4);
			MatrixBlock ret = FederationUtils.aggAdd(tmp);
			ec.setMatrixOutput(output.getName(), ret);
		}
		else { //other combinations
			throw new DMLRuntimeException("Federated AggregateBinary not supported with the "
				+ "following federated objects: "+mo1.isFederated()+":"+mo1.getFedMapping()
				+" "+mo2.isFederated()+":"+mo2.getFedMapping());
		}
	}
}
