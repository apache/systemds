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

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.hops.fedplanner.FTypes.AlignType;
import org.apache.sysds.hops.fedplanner.FTypes.FType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest.RequestType;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse;
import org.apache.sysds.runtime.controlprogram.federated.FederationMap;
import org.apache.sysds.runtime.controlprogram.federated.FederationUtils;
import org.apache.sysds.runtime.controlprogram.federated.MatrixLineagePair;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.AggregateBinaryCPInstruction;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.Operator;

public class AggregateBinaryFEDInstruction extends BinaryFEDInstruction {
	private static final Log LOG = LogFactory.getLog(AggregateBinaryFEDInstruction.class.getName());
	
	public AggregateBinaryFEDInstruction(Operator op, CPOperand in1,
		CPOperand in2, CPOperand out, String opcode, String istr) {
		super(FEDType.AggregateBinary, op, in1, in2, out, opcode, istr);
	}

	public AggregateBinaryFEDInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out,
		String opcode, String istr, FederatedOutput fedOut) {
		super(FEDType.AggregateBinary, op, in1, in2, out, opcode, istr, fedOut);
	}

	public static AggregateBinaryFEDInstruction parseInstruction(AggregateBinaryCPInstruction inst,
		ExecutionContext ec) {
		if(inst.input1.isMatrix() && inst.input2.isMatrix()) {
			MatrixObject mo1 = ec.getMatrixObject(inst.input1);
			MatrixObject mo2 = ec.getMatrixObject(inst.input2);
			if((mo1.isFederated(FType.ROW) && mo1.isFederatedExcept(FType.BROADCAST)) ||
				(mo2.isFederated(FType.ROW) && mo2.isFederatedExcept(FType.BROADCAST)) ||
				(mo1.isFederated(FType.COL) && mo1.isFederatedExcept(FType.BROADCAST))) {
				return AggregateBinaryFEDInstruction.parseInstruction(inst);
			}
		}
		return null;
	}

	private static AggregateBinaryFEDInstruction parseInstruction(AggregateBinaryCPInstruction instr) {
		return new AggregateBinaryFEDInstruction(instr.getOperator(), instr.input1, instr.input2, instr.output,
			instr.getOpcode(), instr.getInstructionString(), FederatedOutput.NONE);
	}

	public static AggregateBinaryFEDInstruction parseInstruction(String str) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		if(!opcode.equalsIgnoreCase(Opcodes.MMULT.toString()))
			throw new DMLRuntimeException("AggregateBinaryInstruction.parseInstruction():: Unknown opcode " + opcode);

		InstructionUtils.checkNumFields(parts, 5);
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand in2 = new CPOperand(parts[2]);
		CPOperand out = new CPOperand(parts[3]);
		int k = Integer.parseInt(parts[4]);
		FederatedOutput fedOut = FederatedOutput.valueOf(parts[5]);
		return new AggregateBinaryFEDInstruction(
			InstructionUtils.getMatMultOperator(k), in1, in2, out, opcode, str, fedOut);
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) {
		MatrixLineagePair mo1 = ec.getMatrixLineagePair(input1);
		MatrixLineagePair mo2 = ec.getMatrixLineagePair(input2);

		//TODO cleanup unnecessary redundancy
		//#1 federated matrix-vector multiplication
		if(mo1.isFederated(FType.COL) && mo2.isFederated(FType.ROW)
			&& mo1.getFedMapping().isAligned(mo2.getFedMapping(), AlignType.COL_T) ) {
			FederatedRequest fr1 = FederationUtils.callInstruction(instString, output,
				new CPOperand[]{input1, input2},
				new long[]{mo1.getFedMapping().getID(), mo2.getFedMapping().getID()}, true);
			if ( _fedOut.isForcedFederated() )
				writeInfoLog(mo1, mo2);
			aggregateLocally(mo1.getFedMapping(), true, ec, fr1);
		}
		else if(mo1.isFederated(FType.ROW)) { // MV + MM
			//construct commands: broadcast rhs, fed mv, retrieve results
			FederatedRequest fr1 = mo1.getFedMapping().broadcast(mo2);
			FederatedRequest fr2 = FederationUtils.callInstruction(instString, output,
				new CPOperand[]{input1, input2},
				new long[]{mo1.getFedMapping().getID(), fr1.getID()}, true);

			boolean isVector = mo2.getNumColumns() == 1;
			boolean isPartOut = mo1.isFederated(FType.PART) || // MV and MM
				(!isVector && mo2.isFederated(FType.PART)); // only MM
			if(isPartOut && _fedOut.isForcedFederated()) {
				writeInfoLog(mo1, mo2);
			}
			if((_fedOut.isForcedFederated() || (!isVector && !_fedOut.isForcedLocal()))
				&& !isPartOut) { // not creating federated output in the MV case for reasons of performance
				Future<FederatedResponse>[] ffr = mo1.getFedMapping().execute(getTID(), true, fr1, fr2);
				setOutputFedMapping(mo1.getFedMapping(), mo1, mo2,
					FederationUtils.sumNonZeros(ffr), fr2.getID(), ec);
			}
			else {
				boolean isDoubleBroadcast = (mo1.isFederated(FType.BROADCAST) && mo2.isFederated(FType.BROADCAST));
				if (isDoubleBroadcast){
					aggregateLocallySingleWorker(mo1.getFedMapping(), ec, fr1, fr2);
				}
				else{
					aggregateLocally(mo1.getFedMapping(), false, ec, fr1, fr2);
				}
			}
		}
		//#2 vector - federated matrix multiplication
		else if (mo2.isFederated(FType.ROW)) {// VM + MM
			//construct commands: broadcast rhs, fed mv, retrieve results
			FederatedRequest[] fr1 = mo2.getFedMapping().broadcastSliced(mo1, true);
			FederatedRequest fr2 = FederationUtils.callInstruction(instString, output,
				new CPOperand[]{input1, input2},
				new long[]{fr1[0].getID(), mo2.getFedMapping().getID()}, true);
			if ( _fedOut.isForcedFederated() ){
				writeInfoLog(mo1, mo2);
			}
			aggregateLocally(mo2.getFedMapping(), true, ec, fr1, fr2);
		}
		//#3 col-federated matrix vector multiplication
		else if (mo1.isFederated(FType.COL)) {// VM + MM
			//construct commands: broadcast rhs, fed mv, retrieve results
			FederatedRequest[] fr1 = mo1.getFedMapping().broadcastSliced(mo2, true);
			FederatedRequest fr2 = FederationUtils.callInstruction(instString, output,
				new CPOperand[]{input1, input2},
				new long[]{mo1.getFedMapping().getID(), fr1[0].getID()}, true);
			if ( _fedOut.isForcedFederated() ){
				writeInfoLog(mo1, mo2);
			}
			aggregateLocally(mo1.getFedMapping(), true, ec, fr1, fr2);
		}
		else { //other combinations
			throw new DMLRuntimeException("Federated AggregateBinary not supported with the "
				+ "following federated objects: "+mo1.isFederated()+":"+mo1.getFedMapping()
				+" "+mo2.isFederated()+":"+mo2.getFedMapping());
		}
	}

	private void writeInfoLog(MatrixLineagePair mo1, MatrixLineagePair mo2){
		FType mo1FType = (mo1.getFedMapping()==null) ? null : mo1.getFedMapping().getType();
		FType mo2FType = (mo2.getFedMapping()==null) ? null : mo2.getFedMapping().getType();
		LOG.info("Federated output flag would result in PART federated map and has been ignored in " + instString);
		LOG.info("Input 1 FType is " + mo1FType + " and input 2 FType " + mo2FType);
	}

	/**
	 * Sets the output with a federated mapping of overlapping partial aggregates.
	 * @param federationMap federated map from which the federated metadata is retrieved
	 * @param mo1 matrix object with number of rows used to set the number of rows of the output
	 * @param mo2 matrix object with number of columns used to set the number of columns of the output
	 * @param outputID ID of the output
	 * @param ec execution context
	 */
	@SuppressWarnings("unused")
	private void setPartialOutput(FederationMap federationMap, MatrixLineagePair mo1, MatrixLineagePair mo2,
		long outputID, ExecutionContext ec){
		MatrixObject out = ec.getMatrixObject(output);
		out.getDataCharacteristics().setDimension(mo1.getNumRows(), mo2.getNumColumns())
			.setBlocksize(mo1.getBlocksize());
		FederationMap outputFedMap = federationMap
			.copyWithNewIDAndRange(mo1.getNumRows(), mo2.getNumColumns(), outputID);
		out.setFedMapping(outputFedMap);
	}

	/**
	 * Sets the output with a federated map copied from federationMap input given.
	 * @param federationMap federation map to be set in output
	 * @param mo1 matrix object with number of rows used to set the number of rows of the output
	 * @param mo2 matrix object with number of columns used to set the number of columns of the output
	 * @param nnz the number of non-zeros of the output
	 * @param outputID ID of the output
	 * @param ec execution context
	 */
	private void setOutputFedMapping(FederationMap federationMap, MatrixLineagePair mo1,
		MatrixLineagePair mo2, long nnz, long outputID, ExecutionContext ec){
		MatrixObject out = ec.getMatrixObject(output);
		out.getDataCharacteristics()
			.setDimension(mo1.getNumRows(), mo2.getNumColumns())
			.setBlocksize(mo1.getBlocksize()).setNonZeros(nnz);
		out.setFedMapping(federationMap.copyWithNewID(outputID, mo2.getNumColumns()));
	}

	private void aggregateLocally(FederationMap fedMap, boolean aggAdd, ExecutionContext ec,
		FederatedRequest... fr) {
		aggregateLocally(fedMap, aggAdd, ec, null, fr);
	}

	/**
	 * Get the partial results and aggregate the partial results locally
	 * @param fedMap the federated mapping
	 * @param aggAdd indicates whether to aggregate the results by addition or binding
	 * @param ec execution context
	 * @param frSliced the federated request array from a sliced broadcast
	 * @param fr the previous federated requests
	 * NOTE: the last federated request fr has to be the instruction call
	 */
	private void aggregateLocally(FederationMap fedMap, boolean aggAdd, ExecutionContext ec,
		FederatedRequest[] frSliced, FederatedRequest... fr) {
		long callInstID = fr[fr.length - 1].getID();
		FederatedRequest frG = new FederatedRequest(RequestType.GET_VAR, callInstID);
		FederatedRequest frC = fedMap.cleanup(getTID(), callInstID);
		//execute federated operations and aggregate
		Future<FederatedResponse>[] ffr;
		if(frSliced != null)
			ffr = fedMap.execute(getTID(), frSliced, ArrayUtils.addAll(fr, frG, frC));
		else
			ffr = fedMap.execute(getTID(), ArrayUtils.addAll(fr, frG, frC));

		MatrixBlock ret;
		if ( aggAdd )
			ret = FederationUtils.aggAdd(ffr);
		else
			ret = FederationUtils.bind(ffr, false);
		ec.setMatrixOutput(output.getName(), ret);
	}

	private void aggregateLocallySingleWorker(FederationMap fedMap, ExecutionContext ec, FederatedRequest... fr) {
		//create GET calls on output
		long callInstID = fr[fr.length - 1].getID();
		FederatedRequest frG = new FederatedRequest(RequestType.GET_VAR, callInstID);
		FederatedRequest frC = fedMap.cleanup(getTID(), callInstID);
		//execute federated operations
		Future<FederatedResponse>[] ffr = fedMap.execute(getTID(), ArrayUtils.addAll(fr, frG, frC));
		try {
			//use only one response (all responses contain the same result)
			MatrixBlock ret = (MatrixBlock) ffr[0].get().getData()[0];
			ec.setMatrixOutput(output.getName(), ret);
		} catch(Exception ex){
			throw new DMLRuntimeException(ex);
		}
	}
}
