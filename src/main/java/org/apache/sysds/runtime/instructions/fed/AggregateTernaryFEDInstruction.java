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
import org.apache.sysds.hops.fedplanner.FTypes.AlignType;
import org.apache.sysds.hops.fedplanner.FTypes.FType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest.RequestType;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse;
import org.apache.sysds.runtime.controlprogram.federated.FederationUtils;
import org.apache.sysds.runtime.controlprogram.federated.MatrixLineagePair;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.AggregateTernaryCPInstruction;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.DoubleObject;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.instructions.spark.AggregateTernarySPInstruction;
import org.apache.sysds.runtime.matrix.operators.AggregateTernaryOperator;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysds.runtime.matrix.operators.Operator;

public class AggregateTernaryFEDInstruction extends ComputationFEDInstruction {
	// private static final Log LOG = LogFactory.getLog(AggregateTernaryFEDInstruction.class.getName());

	private AggregateTernaryFEDInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out,
		String opcode, String istr, FederatedOutput fedOut) {
		super(FEDType.AggregateTernary, op, in1, in2, in3, out, opcode, istr, fedOut);
	}

	public static AggregateTernaryFEDInstruction parseInstruction(AggregateTernaryCPInstruction inst,
		ExecutionContext ec) {
		if(inst.input1.isMatrix() && ec.getCacheableData(inst.input1).isFederatedExcept(FType.BROADCAST) &&
			inst.input2.isMatrix() && ec.getCacheableData(inst.input2).isFederatedExcept(FType.BROADCAST)) {
			return parseInstruction(inst);
		}
		return null;
	}

	public static AggregateTernaryFEDInstruction parseInstruction(AggregateTernarySPInstruction inst,
		ExecutionContext ec) {
		if(inst.input1.isMatrix() && ec.getCacheableData(inst.input1).isFederatedExcept(FType.BROADCAST) &&
			inst.input2.isMatrix() && ec.getCacheableData(inst.input2).isFederatedExcept(FType.BROADCAST)) {
			return parseInstruction(inst);
		}
		return null;
	}

	private static AggregateTernaryFEDInstruction parseInstruction(AggregateTernaryCPInstruction instr) {
		return new AggregateTernaryFEDInstruction(instr.getOperator(), instr.input1, instr.input2, instr.input3,
			instr.output, instr.getOpcode(), instr.getInstructionString(), FederatedOutput.NONE);
	}

	private static AggregateTernaryFEDInstruction parseInstruction(AggregateTernarySPInstruction instr) {
		return new AggregateTernaryFEDInstruction(instr.getOperator(), instr.input1, instr.input2, instr.input3,
			instr.output, instr.getOpcode(), instr.getInstructionString(), FederatedOutput.NONE);
	}

	public static AggregateTernaryFEDInstruction parseInstruction(String str) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];

		if(opcode.equalsIgnoreCase(Opcodes.TAKPM.toString()) || opcode.equalsIgnoreCase(Opcodes.TACKPM.toString())) {
			InstructionUtils.checkNumFields(parts, 5, 6);

			CPOperand in1 = new CPOperand(parts[1]);
			CPOperand in2 = new CPOperand(parts[2]);
			CPOperand in3 = new CPOperand(parts[3]);
			CPOperand out = new CPOperand(parts[4]);
			int numThreads = Integer.parseInt(parts[5]);
			FederatedOutput fedOut = FederatedOutput.NONE;
			if ( parts.length == 7 )
				fedOut = FederatedOutput.valueOf(parts[6]);

			AggregateTernaryOperator op = InstructionUtils.parseAggregateTernaryOperator(opcode, numThreads);
			return new AggregateTernaryFEDInstruction(op, in1, in2, in3, out, opcode, str, fedOut);
		}
		throw new DMLRuntimeException("AggregateTernaryInstruction.parseInstruction():: Unknown opcode " + opcode);
}

	@Override
	public void processInstruction(ExecutionContext ec) {
		MatrixLineagePair mo1 = ec.getMatrixLineagePair(input1);
		MatrixLineagePair mo2 = ec.getMatrixLineagePair(input2);
		MatrixLineagePair mo3 = input3.isLiteral() ? null : ec.getMatrixLineagePair(input3);
		if(mo3 != null && mo1.isFederated() && mo2.isFederated() && mo3.isFederated()
				&& mo1.getFedMapping().isAligned(mo2.getFedMapping(), mo1.isFederated(FType.ROW) ? AlignType.ROW : AlignType.COL)
				&& mo2.getFedMapping().isAligned(mo3.getFedMapping(), mo1.isFederated(FType.ROW) ? AlignType.ROW : AlignType.COL)) {
			FederatedRequest fr1 = FederationUtils.callInstruction(getInstructionString(), output,
				new CPOperand[] {input1, input2, input3},
				new long[] {mo1.getFedMapping().getID(), mo2.getFedMapping().getID(), mo3.getFedMapping().getID()},
				true);
			FederatedRequest fr2 = new FederatedRequest(RequestType.GET_VAR, fr1.getID());
			FederatedRequest fr3 = mo1.getFedMapping().cleanup(getTID(), fr1.getID());
			Future<FederatedResponse>[] response = mo1.getFedMapping().execute(getTID(), fr1, fr2, fr3);
			
			if(output.getDataType().isScalar()) {
				AggregateUnaryOperator aop = InstructionUtils.parseBasicAggregateUnaryOperator("uak+");
				ec.setScalarOutput(output.getName(), FederationUtils.aggScalar(aop, response, mo1.getFedMapping()));
			}
			else {
				AggregateUnaryOperator aop = InstructionUtils.parseBasicAggregateUnaryOperator(getOpcode().equals("fed_tak+*") ? Opcodes.UAKP.toString() : Opcodes.UACKP.toString());
				ec.setMatrixOutput(output.getName(), FederationUtils.aggMatrix(aop, response, mo1.getFedMapping()));
			}
		}
		else if(mo1.isFederated() && mo2.isFederated()
			&& mo1.getFedMapping().isAligned(mo2.getFedMapping(), false)) {
			FederatedRequest[] fr1 = (mo3 == null) ?
				new FederatedRequest[] {mo1.getFedMapping().broadcast(ec.getScalarInput(input3))} :
				mo1.getFedMapping().broadcastSliced(mo3, false);
			FederatedRequest fr2 = FederationUtils.callInstruction(instString, output,
				new CPOperand[] {input1, input2, input3},
				new long[] {mo1.getFedMapping().getID(), mo2.getFedMapping().getID(), fr1[0].getID()}, true);
			FederatedRequest fr3 = new FederatedRequest(RequestType.GET_VAR, fr2.getID());
			FederatedRequest fr4 = (mo3 == null) ? 
				mo2.getFedMapping().cleanup(getTID(), fr1[0].getID(), fr2.getID()) :
				mo2.getFedMapping().cleanup(getTID(), fr2.getID()); //no cleanup of broadcasts
			Future<FederatedResponse>[] tmp = (mo3 == null) ?
				mo1.getFedMapping().execute(getTID(), fr1[0], fr2, fr3, fr4) :
				mo1.getFedMapping().execute(getTID(), fr1, fr2, fr3, fr4);

			if(output.getDataType().isScalar()) {
				double sum = 0;
				for(Future<FederatedResponse> fr : tmp)
					try {
						sum += ((ScalarObject) fr.get().getData()[0]).getDoubleValue();
					}
					catch(Exception e) {
						throw new DMLRuntimeException("Federated Get data failed with exception on TernaryFedInstruction", e);
					}
				ec.setScalarOutput(output.getName(), new DoubleObject(sum));
			}
			else {
				throw new DMLRuntimeException("Not Implemented Federated Ternary Variation");
			}
		} else if(mo1.isFederatedExcept(FType.BROADCAST) && input3.isMatrix() && mo3 != null) {
			FederatedRequest[] fr1 = mo1.getFedMapping().broadcastSliced(mo3, false);
			FederatedRequest[] fr2 = mo1.getFedMapping().broadcastSliced(mo2, false);
			FederatedRequest fr3 = FederationUtils.callInstruction(getInstructionString(), output,
				new CPOperand[] {input1, input2, input3},
				new long[] {mo1.getFedMapping().getID(), fr2[0].getID(), fr1[0].getID()}, true);
			FederatedRequest fr4 = new FederatedRequest(RequestType.GET_VAR, fr3.getID());

			FederatedRequest[][] frSlices = new FederatedRequest[][]{fr1,fr2};
			FederatedRequest[] frProcessAndGet = new FederatedRequest[]{fr3,fr4};
			Future<FederatedResponse>[] tmp = mo1.getFedMapping()
				.executeMultipleSlices(getTID(), true, frSlices, frProcessAndGet);

			if(output.getDataType().isScalar()) {
				double sum = 0;
				for(Future<FederatedResponse> fr : tmp)
					try {
						sum += ((ScalarObject) fr.get().getData()[0]).getDoubleValue();
					}
					catch(Exception e) {
						throw new DMLRuntimeException("Federated Get data failed with exception on TernaryFedInstruction", e);
					}
				ec.setScalarOutput(output.getName(), new DoubleObject(sum));
			}
			else {
				throw new DMLRuntimeException("Not Implemented Federated Ternary Variation");
			}
		}
		else {
			if(mo3 == null)
				throw new DMLRuntimeException("Federated AggregateTernary not supported with the "
					+ "following federated objects: " + mo1.isFederated() + ":" + mo1.getFedMapping() + " "
					+ mo2.isFederated() + ":" + mo2.getFedMapping());
			throw new DMLRuntimeException("Federated AggregateTernary not supported with the "
				+ "following federated objects: " + mo1.isFederated() + ":" + mo1.getFedMapping() + " "
				+ mo2.isFederated() + ":" + mo2.getFedMapping() + mo3.isFederated() + ":" + mo3.getFedMapping());
		}
	}
}
