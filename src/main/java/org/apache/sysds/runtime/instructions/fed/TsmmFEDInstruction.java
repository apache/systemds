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

import java.util.concurrent.Future;

import org.apache.sysds.common.Opcodes;
import org.apache.sysds.hops.fedplanner.FTypes.FType;
import org.apache.sysds.lops.MMTSJ.MMTSJType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest.RequestType;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse;
import org.apache.sysds.runtime.controlprogram.federated.FederationMap;
import org.apache.sysds.runtime.controlprogram.federated.FederationUtils;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.MMTSJCPInstruction;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public class TsmmFEDInstruction extends BinaryFEDInstruction {
	private final MMTSJType _type;
	@SuppressWarnings("unused")
	private final int _numThreads;
	
	public TsmmFEDInstruction(CPOperand in, CPOperand out, MMTSJType type, int k, String opcode, String istr, FederatedOutput fedOut) {
		super(FEDType.Tsmm, null, in, null, out, opcode, istr, fedOut);
		_type = type;
		_numThreads = k;
	}

	public TsmmFEDInstruction(CPOperand in, CPOperand out, MMTSJType type, int k, String opcode, String istr) {
		this(in, out, type, k, opcode, istr, FederatedOutput.NONE);
	}

	public static TsmmFEDInstruction parseInstruction(MMTSJCPInstruction inst, ExecutionContext ec) {
		MatrixObject mo = ec.getMatrixObject(inst.input1);
		if( (mo.isFederated(FType.ROW) && mo.isFederatedExcept(FType.BROADCAST) && inst.getMMTSJType().isLeft()) ||
			(mo.isFederated(FType.COL) && mo.isFederatedExcept(FType.BROADCAST) && inst.getMMTSJType().isRight()))
			return  parseInstruction(inst);
		return null;
	}	

	private static TsmmFEDInstruction parseInstruction(MMTSJCPInstruction instr) {
		return new TsmmFEDInstruction(instr.input1, instr.getOutput(), instr.getMMTSJType(), instr.getNumThreads(),
			instr.getOpcode(), instr.getInstructionString());
	}

	public static TsmmFEDInstruction parseInstruction(String str) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		if(!opcode.equalsIgnoreCase(Opcodes.TSMM.toString()))
			throw new DMLRuntimeException("TsmmFedInstruction.parseInstruction():: Unknown opcode " + opcode);

		InstructionUtils.checkNumFields(parts, 3, 4, 5);
		CPOperand in = new CPOperand(parts[1]);
		CPOperand out = new CPOperand(parts[2]);
		MMTSJType type = MMTSJType.valueOf(parts[3]);
		int k = (parts.length > 4) ? Integer.parseInt(parts[4]) : -1;
		FederatedOutput fedOut = (parts.length > 5) ? FederatedOutput.valueOf(parts[5]) : FederatedOutput.NONE;
		return new TsmmFEDInstruction(in, out, type, k, opcode, str, fedOut);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		MatrixObject mo1 = ec.getMatrixObject(input1);
		if((_type.isLeft() && mo1.isFederated(FType.ROW)) || (mo1.isFederated(FType.COL) && _type.isRight()))
			processRowCol(ec, mo1);
		else { //other combinations
			String exMessage = (!mo1.isFederated() || mo1.getFedMapping() == null) ?
				"Federated Tsmm does not support non-federated input" :
				"Federated Tsmm does not support federated map type " + mo1.getFedMapping().getType();
			throw new DMLRuntimeException(exMessage);
		}
	}

	private void processRowCol(ExecutionContext ec, MatrixObject mo1){
		FederatedRequest fr1 = FederationUtils.callInstruction(instString, output,
			new CPOperand[]{input1}, new long[]{mo1.getFedMapping().getID()}, true);
		if (_fedOut.isForcedFederated()){
			fr1 = mo1.getFedMapping().broadcast(mo1);
			FederatedRequest fr2 = FederationUtils.callInstruction(instString, output,
				new CPOperand[]{input1}, new long[]{fr1.getID()}, true);
			mo1.getFedMapping().execute(getTID(), fr1, fr2);
			setOutputFederated(ec, mo1, fr2, FType.BROADCAST);
		}
		else if (mo1.isFederated(FType.BROADCAST)){
			FederatedRequest fr2 = new FederatedRequest(RequestType.GET_VAR, fr1.getID());
			Future<FederatedResponse>[] tmp = mo1.getFedMapping().execute(getTID(), fr1, fr2);
			MatrixBlock[] outBlocks = FederationUtils.getResults(tmp);
			ec.setMatrixOutput(output.getName(), outBlocks[0]);
		}
		else {
			FederatedRequest fr2 = new FederatedRequest(RequestType.GET_VAR, fr1.getID());
			FederatedRequest fr3 = mo1.getFedMapping().cleanup(getTID(), fr1.getID());

			//execute federated operations and aggregate
			Future<FederatedResponse>[] tmp = mo1.getFedMapping().execute(getTID(), fr1, fr2, fr3);
			MatrixBlock ret = FederationUtils.aggAdd(tmp);
			ec.setMatrixOutput(output.getName(), ret);
		}
	}

	private void setOutputFederated(ExecutionContext ec, MatrixObject mo1, FederatedRequest fr1, FType outFType){
		MatrixObject out = ec.getMatrixObject(output);
		out.getDataCharacteristics()
			.set(mo1.getNumColumns(), mo1.getNumColumns(), mo1.getBlocksize());
		FederationMap outputFedMap = mo1.getFedMapping()
			.copyWithNewIDAndRange(mo1.getNumColumns(), mo1.getNumColumns(), fr1.getID(), outFType);
		out.setFedMapping(outputFedMap);
	}
}
