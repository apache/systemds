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

import java.util.Arrays;
import java.util.Objects;
import java.util.concurrent.Future;

import org.apache.sysds.common.Types;
import org.apache.sysds.hops.fedplanner.FTypes.FType;
import org.apache.sysds.runtime.controlprogram.caching.FrameObject;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse;
import org.apache.sysds.runtime.controlprogram.federated.FederationUtils;
import org.apache.sysds.runtime.controlprogram.federated.MatrixLineagePair;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.TernaryCPInstruction;
import org.apache.sysds.runtime.instructions.cp.TernaryFrameScalarCPInstruction;
import org.apache.sysds.runtime.instructions.spark.TernaryFrameScalarSPInstruction;
import org.apache.sysds.runtime.instructions.spark.TernarySPInstruction;
import org.apache.sysds.runtime.matrix.operators.TernaryOperator;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;


public class TernaryFEDInstruction extends ComputationFEDInstruction {

	protected TernaryFEDInstruction(TernaryOperator op, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out,
		String opcode, String str, FederatedOutput fedOut) {
		super(FEDInstruction.FEDType.Ternary, op, in1, in2, in3, out, opcode, str, fedOut);
	}

	public static TernaryFEDInstruction parseInstruction(TernaryCPInstruction inst, ExecutionContext ec) {
		if(inst.getOpcode().equals("_map") && inst instanceof TernaryFrameScalarCPInstruction &&
			!inst.getInstructionString().contains("UtilFunctions") && inst.input1.isFrame() &&
			ec.getFrameObject(inst.input1).isFederated()) {
			long margin = ec.getScalarInput(inst.input3).getLongValue();
			FrameObject fo = ec.getFrameObject(inst.input1);
			if(margin == 0 || (fo.isFederated(FType.ROW) && margin == 1) || (fo.isFederated(FType.COL) && margin == 2))
				return TernaryFrameScalarFEDInstruction.parseInstruction((TernaryFrameScalarCPInstruction) inst);
		}
		else if((inst.input1.isMatrix() && ec.getCacheableData(inst.input1).isFederatedExcept(FType.BROADCAST)) ||
			(inst.input2.isMatrix() && ec.getCacheableData(inst.input2).isFederatedExcept(FType.BROADCAST)) ||
			(inst.input3.isMatrix() && ec.getCacheableData(inst.input3).isFederatedExcept(FType.BROADCAST))) {
			return TernaryFEDInstruction.parseInstruction(inst);
		}
		return null;
	}

	public static TernaryFEDInstruction parseInstruction(TernarySPInstruction inst, ExecutionContext ec) {
		if(inst.getOpcode().equals("_map") && inst instanceof TernaryFrameScalarSPInstruction &&
			!inst.getInstructionString().contains("UtilFunctions") && inst.input1.isFrame() &&
			ec.getFrameObject(inst.input1).isFederated()) {
			long margin = ec.getScalarInput(inst.input3).getLongValue();
			FrameObject fo = ec.getFrameObject(inst.input1);
			if(margin == 0 || (fo.isFederated(FType.ROW) && margin == 1) || (fo.isFederated(FType.COL) && margin == 2))
				return TernaryFrameScalarFEDInstruction.parseInstruction((TernaryFrameScalarSPInstruction) inst);
		}
		else if((inst.input1.isMatrix() && ec.getCacheableData(inst.input1).isFederatedExcept(FType.BROADCAST)) ||
			(inst.input2.isMatrix() && ec.getCacheableData(inst.input2).isFederatedExcept(FType.BROADCAST)) ||
			(inst.input3.isMatrix() && ec.getCacheableData(inst.input3).isFederatedExcept(FType.BROADCAST))) {
			return TernaryFEDInstruction.parseInstruction(inst);
		}
		return null;
	}

	private static TernaryFEDInstruction parseInstruction(TernaryCPInstruction instr) {
		return new TernaryFEDInstruction((TernaryOperator) instr.getOperator(), instr.input1, instr.input2,
			instr.input3, instr.output, instr.getOpcode(), instr.getInstructionString(), FederatedOutput.NONE);
	}

	private static TernaryFEDInstruction parseInstruction(TernarySPInstruction instr) {
		return new TernaryFEDInstruction((TernaryOperator) instr.getOperator(), instr.input1, instr.input2,
				instr.input3, instr.output, instr.getOpcode(), instr.getInstructionString(), FederatedOutput.NONE);
	}

	public static TernaryFEDInstruction parseInstruction(String str) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		CPOperand operand1 = new CPOperand(parts[1]);
		CPOperand operand2 = new CPOperand(parts[2]);
		CPOperand operand3 = new CPOperand(parts[3]);
		CPOperand outOperand = new CPOperand(parts[4]);
		int numThreads = parts.length > 5 & !opcode.contains("map") ? Integer.parseInt(parts[5]) : 1;
		FederatedOutput fedOut = parts.length >= 7 && !opcode.contains("map") ? FederatedOutput
			.valueOf(parts[6]) : FederatedOutput.NONE;
		TernaryOperator op = InstructionUtils.parseTernaryOperator(opcode, numThreads);
		if(operand1.isFrame() && operand2.isScalar() || operand2.isFrame() && operand1.isScalar())
			return new TernaryFrameScalarFEDInstruction(op, operand1, operand2, operand3, outOperand, opcode,
				InstructionUtils.removeFEDOutputFlag(str), fedOut);
		return new TernaryFEDInstruction(op, operand1, operand2, operand3, outOperand, opcode, str, fedOut);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		MatrixLineagePair mo1 = input1.isMatrix() ? ec.getMatrixLineagePair(input1) : null;
		MatrixLineagePair mo2 = input2.isMatrix() ? ec.getMatrixLineagePair(input2) : null;
		MatrixLineagePair mo3 = input3 != null && input3.isMatrix() ? ec.getMatrixLineagePair(input3) : null;

		long matrixInputsCount = Arrays.asList(mo1, mo2, mo3)
			.stream().filter(Objects::nonNull).count();

		if(matrixInputsCount == 3)
			processMatrixInput(ec, mo1, mo2, mo3);
		else if(matrixInputsCount == 1) {
			CPOperand in;
			// determine the position of a matrix in the input and whether any of the scalars are not literals
			if (mo1 == null) {
				if (mo2 == null) { // sc, sc, mat
					in = input3;
					instString = InstructionUtils.replaceOperand(instString, 2,
						InstructionUtils.createLiteralOperand(ec.getScalarInput(input1).getStringValue(), Types.ValueType.FP64));
					if (!input2.isLiteral()) {
						instString = InstructionUtils.replaceOperand(instString, 3,
							InstructionUtils.createLiteralOperand(ec.getScalarInput(input2).getStringValue(), Types.ValueType.FP64));
					}
				} else { // sc, mat, sc
					in = input2;
					instString = InstructionUtils.replaceOperand(instString, 2,
						InstructionUtils.createLiteralOperand(ec.getScalarInput(input1).getStringValue(), Types.ValueType.FP64));
					if (!input3.isLiteral()) {
						instString = InstructionUtils.replaceOperand(instString, 4,
							InstructionUtils.createLiteralOperand(ec.getScalarInput(input3).getStringValue(), Types.ValueType.FP64));
					}
				}
			} else { // mat, sc, sc
				in = input1;
				if (!input2.isLiteral()) {
					instString = InstructionUtils.replaceOperand(instString, 3,
						InstructionUtils.createLiteralOperand(ec.getScalarInput(input2).getStringValue(), Types.ValueType.FP64));
				}
				if (!input3.isLiteral()) {
					instString = InstructionUtils.replaceOperand(instString, 4,
						InstructionUtils.createLiteralOperand(ec.getScalarInput(input3).getStringValue(), Types.ValueType.FP64));
				}
			}

			mo1 = mo1 == null ? mo2 == null ? mo3 : mo2 : mo1;
			processMatrixScalarInput(ec, mo1, in);
		}
		else {
			if(mo1 != null && mo2 != null) {
				if(input3 != null && !input3.isLiteral())
					instString = InstructionUtils.replaceOperand(instString, 4,
						InstructionUtils.createLiteralOperand(ec.getScalarInput(input3).getStringValue(), Types.ValueType.FP64));
				process2MatrixScalarInput(ec, mo1, mo2, input1, input2);
			}
			else if(mo2 != null && mo3 != null) {
				if(!input1.isLiteral())
					instString = InstructionUtils.replaceOperand(instString, 2,
						InstructionUtils.createLiteralOperand(ec.getScalarInput(input1).getStringValue(), Types.ValueType.FP64));
				process2MatrixScalarInput(ec, mo2, mo3, input2, input3);
			}
			else if(mo1 != null && mo3 != null) {
				if(!input2.isLiteral())
					instString = InstructionUtils.replaceOperand(instString, 3,
						InstructionUtils.createLiteralOperand(ec.getScalarInput(input2).getStringValue(), Types.ValueType.FP64));
				process2MatrixScalarInput(ec, mo1, mo3, input1, input3);
			}
		}
	}

	private void processMatrixScalarInput(ExecutionContext ec, MatrixLineagePair mo1, CPOperand in) {
		long id = FederationUtils.getNextFedDataID();
		FederatedRequest fr = FederationUtils.callInstruction(instString, output, id, new CPOperand[] {in}, new long[] {mo1.getFedMapping().getID()},
			InstructionUtils.getExecType(instString), false);

		sendFederatedRequests(ec, mo1.getMO(), fr.getID(), fr);
	}

	private void process2MatrixScalarInput(ExecutionContext ec, MatrixLineagePair mo1, MatrixLineagePair mo2, CPOperand in1, CPOperand in2) {
		FederatedRequest[] fr1 = null;
		CPOperand[] varOldIn;
		long[] varNewIn;
		varOldIn = new CPOperand[] {in1, in2};
		if(mo1.isFederated()) {
			if(mo2.isFederated() && mo1.getFedMapping().isAligned(mo2.getFedMapping(), false))
				varNewIn = new long[]{mo1.getFedMapping().getID(), mo2.getFedMapping().getID()};
			else {
				fr1 = mo1.getFedMapping().broadcastSliced(mo2, false);
				varNewIn = new long[]{mo1.getFedMapping().getID(), fr1[0].getID()};
			}
		} else {
			mo1 = ec.getMatrixLineagePair(in2);
			fr1 = mo1.getFedMapping().broadcastSliced(ec.getMatrixLineagePair(in1), false);
			varNewIn = new long[]{fr1[0].getID(), mo1.getFedMapping().getID()};
		}
		long id = FederationUtils.getNextFedDataID();
		Types.ExecType execType = InstructionUtils.getExecType(instString) == Types.ExecType.SPARK ? Types.ExecType.SPARK : Types.ExecType.CP;
		FederatedRequest fr2 = new FederatedRequest(FederatedRequest.RequestType.PUT_VAR, id, new MatrixCharacteristics(-1, -1), mo1.getDataType());
		FederatedRequest fr3 = FederationUtils.callInstruction(instString, output, id, varOldIn, varNewIn, execType, false);

		// 2 aligned inputs
		if(fr1 == null)
			sendFederatedRequests(ec, mo1.getMO(), fr3.getID(), fr2, fr3);
		else
			sendFederatedRequests(ec, mo1.getMO(), fr3.getID(), fr1, fr2, fr3);
	}

	/**
	 * Send federated requests and retrieve output if federated output flag is set.
	 * @param ec execution context
	 * @param fedMapObj matrix object with federated mapping where federated requests are sent to.
	 * @param fedOutputID ID of federated output
	 * @param federatedRequests federated requests for processing instruction
	 */
	private void sendFederatedRequests(ExecutionContext ec, MatrixObject fedMapObj, long fedOutputID,
		FederatedRequest... federatedRequests){
		sendFederatedRequests(ec, fedMapObj, fedOutputID, null, null, federatedRequests);
	}

	/**
	 * Send federated requests and retrieve output if federated output flag is set.
	 * @param ec execution context
	 * @param fedMapObj matrix object with federated mapping where federated requests are sent to.
	 * @param fedOutputID ID of federated output
	 * @param federatedSlices federated requests for broadcasting slices before processing instruction
	 * @param federatedRequests federated requests for processing instruction
	 */
	private void sendFederatedRequests(ExecutionContext ec, MatrixObject fedMapObj, long fedOutputID,
		FederatedRequest[] federatedSlices, FederatedRequest... federatedRequests){
		sendFederatedRequests(ec, fedMapObj, fedOutputID, federatedSlices, null, federatedRequests);
	}

	/**
	 * Send federated requests and retrieve output if federated output flag is set.
	 * @param ec execution context
	 * @param fedMapObj matrix object with federated mapping where federated requests are sent to.
	 * @param fedOutputID ID of federated output
	 * @param federatedSlices1 federated requests for broadcasting slices before processing instruction
	 * @param federatedSlices2 federated requests for broadcasting slices before processing instruction
	 * @param federatedRequests federated requests for processing instruction
	 */
	private void sendFederatedRequests(ExecutionContext ec, MatrixObject fedMapObj, long fedOutputID,
		FederatedRequest[] federatedSlices1, FederatedRequest[] federatedSlices2, FederatedRequest... federatedRequests){
		if ( !_fedOut.isForcedLocal() ){
			fedMapObj.getFedMapping().execute(getTID(), true, federatedSlices1, federatedSlices2, federatedRequests);
			setOutputFedMapping(ec, fedMapObj, fedOutputID);
		}
		else
			processAndRetrieve(ec, fedMapObj, fedOutputID, federatedSlices1, federatedSlices2, federatedRequests);
	}

	/**
	 * Process instruction and get output from federated workers.
	 * @param ec execution context
	 * @param fedMapObj matrix object with federated mapping where federated requests are sent to.
	 * @param fedOutputID ID of federated output
	 * @param federatedSlices1 federated requests for broadcasting slices before processing instruction
	 * @param federatedSlices2 federated requests for broadcasting slices before processing instruction
	 * @param federatedRequests federated requests for processing instruction
	 */
	private void processAndRetrieve(ExecutionContext ec, MatrixObject fedMapObj, long fedOutputID,
		FederatedRequest[] federatedSlices1, FederatedRequest[] federatedSlices2, FederatedRequest... federatedRequests){
		FederatedRequest getRequest = new FederatedRequest(FederatedRequest.RequestType.GET_VAR, fedOutputID);
		Future<FederatedResponse>[] executionResponse = fedMapObj.getFedMapping().execute(
			getTID(), true, federatedSlices1, federatedSlices2, collectRequests(federatedRequests, getRequest));
		ec.setMatrixOutput(output.getName(), FederationUtils.bind(executionResponse,
			fedMapObj.isFederated(FType.COL)));
	}

	/**
	 * Collect federated requests into a single array of federated requests.
	 * The federated requests are added in the same order as the parameters of this method.
	 * @param fedRequests array of federated requests
	 * @param fedRequest1 federated request to occur after array
	 * @return federated requests collected in a single array
	 */
	private static FederatedRequest[] collectRequests(FederatedRequest[] fedRequests, FederatedRequest fedRequest1){
		FederatedRequest[] allRequests = new FederatedRequest[fedRequests.length + 1];
		for ( int i = 0; i < fedRequests.length; i++ )
			allRequests[i] = fedRequests[i];
		allRequests[allRequests.length-1] = fedRequest1;
		return allRequests;
	}

	private void processMatrixInput(ExecutionContext ec, MatrixLineagePair mo1, MatrixLineagePair mo2, MatrixLineagePair mo3) {

		// check aligned matrices
		RetAlignedValues retAlignedValues = getAlignedInputs(ec, mo1, mo2, mo3);

		FederatedRequest[] fr2;
		FederatedRequest fr3, fr4;

		long id = FederationUtils.getNextFedDataID();
		FederatedRequest fr5 = new FederatedRequest(FederatedRequest.RequestType.PUT_VAR, id, new MatrixCharacteristics(-1, -1), mo1.getDataType());
		Types.ExecType execType = InstructionUtils.getExecType(instString);

		// all 3 inputs fed aligned on the one worker
		if(retAlignedValues._allAligned) {
			fr3 = FederationUtils.callInstruction(instString, output, id, new CPOperand[] {input1, input2, input3},
				new long[] {mo1.getFedMapping().getID(), mo2.getFedMapping().getID(), mo3.getFedMapping().getID()}, execType, false);
			sendFederatedRequests(ec, mo1.getMO(), fr3.getID(), fr5, fr3);
		}
		// 2 fed aligned inputs
		else if(retAlignedValues._twoAligned) {
			fr3 = FederationUtils.callInstruction(instString, output, id, new CPOperand[] {input1, input2, input3}, retAlignedValues._vars, execType, false);
			fr4 = mo1.getFedMapping().cleanup(getTID(), retAlignedValues._fr[0].getID());
			sendFederatedRequests(ec, mo1.getMO(), fr3.getID(), retAlignedValues._fr, fr5, fr3, fr4);
		}
		// 1 fed input or not aligned
		else {
			if(!mo1.isFederated())
				if(mo2.isFederated()) {
					mo1 = mo2;
					mo2 = ec.getMatrixLineagePair(input1);
				}
				else {
					mo1 = mo3;
					mo3 = ec.getMatrixLineagePair(input1);
				}

			FederatedRequest[] fr1 = mo1.getFedMapping().broadcastSliced(mo2, false);
			fr2 = mo1.getFedMapping().broadcastSliced(mo3, false);

			long[] vars = new long[] {mo1.getFedMapping().getID(), fr1[0].getID(), fr2[0].getID()};
			if(!ec.getMatrixObject(input1).isFederated())
				vars = ec.getMatrixObject(input2).isFederated() ? new long[] {fr1[0].getID(), mo1.getFedMapping().getID(), fr2[0].getID()} : new long[] {fr1[0].getID(), fr2[0].getID(),
					mo1.getFedMapping().getID()};

			fr3 = FederationUtils.callInstruction(instString, output, id, new CPOperand[] {input1, input2, input3}, vars, execType, false);
			sendFederatedRequests(ec, mo1.getMO(), fr3.getID(), fr5, fr1[0], fr2[0], fr3);
		}
	}

	/**
	 * Check alignment of matrices and return aligned federated data.
	 * @param ec execution context
	 * @param mo1 first input matrix
	 * @param mo2 second input matrix
	 * @param mo3 third input matrix
	 * @return aligned federated data
	 */
	private RetAlignedValues getAlignedInputs(ExecutionContext ec, MatrixLineagePair mo1, MatrixLineagePair mo2, MatrixLineagePair mo3) {
		long[] vars = new long[0];
		FederatedRequest[] fr = new FederatedRequest[0];
		boolean allAligned = mo1.isFederated() && mo2.isFederated() && mo3.isFederated() && mo1.getFedMapping().isAligned(mo2.getFedMapping(), false) &&
			mo1.getFedMapping().isAligned(mo3.getFedMapping(), false);
		boolean twoAligned = false;
		if(!allAligned && mo1.isFederated() && !mo1.isFederated(FType.BROADCAST) && mo2.isFederated() &&
			mo1.getFedMapping().isAligned(mo2.getFedMapping(), false)) {
			twoAligned = true;
			fr = mo1.getFedMapping().broadcastSliced(mo3, false);
			vars = new long[] {mo1.getFedMapping().getID(), mo2.getFedMapping().getID(), fr[0].getID()};
		} else if(!allAligned && mo1.isFederated() && !mo1.isFederated(FType.BROADCAST) &&
			mo3.isFederated() && mo1.getFedMapping().isAligned(mo3.getFedMapping(), false)) {
			twoAligned = true;
			fr = mo1.getFedMapping().broadcastSliced(mo2, false);
			vars = new long[] {mo1.getFedMapping().getID(), fr[0].getID(), mo3.getFedMapping().getID()};
		} else if(!mo1.isFederated(FType.BROADCAST) && mo2.isFederated() && mo3.isFederated() && mo2.getFedMapping().isAligned(mo3.getFedMapping(), false) && !allAligned) {
			twoAligned = true;
			mo1 = mo2;
			mo2 = mo3;
			mo3 = ec.getMatrixLineagePair(input1);
			fr = mo1.getFedMapping().broadcastSliced(mo3, false);
			vars = new long[] {fr[0].getID(), mo1.getFedMapping().getID(), mo2.getFedMapping().getID()};
		}

		return new RetAlignedValues(twoAligned, allAligned, vars, fr);
	}

	private static final class RetAlignedValues {
		public boolean _twoAligned;
		public boolean _allAligned;
		public long[] _vars;
		public FederatedRequest[] _fr;

		public RetAlignedValues(boolean twoAligned, boolean allAligned, long[] vars, FederatedRequest[] fr) {
			_twoAligned = twoAligned;
			_allAligned = allAligned;
			_vars = vars;
			_fr = fr;
		}
	}

	/**
	 * Set fed mapping of output. The data characteristics are not set.
	 * @param ec execution context
	 * @param fedMapObj federated matrix object from which federated mapping is derived
	 * @param fedOutputID ID for the fed mapping of output
	 */
	private void setOutputFedMapping(ExecutionContext ec, MatrixObject fedMapObj, long fedOutputID) {
		MatrixObject out = ec.getMatrixObject(output);
		out.setFedMapping(fedMapObj.getFedMapping().copyWithNewID(fedOutputID));
	}
}
