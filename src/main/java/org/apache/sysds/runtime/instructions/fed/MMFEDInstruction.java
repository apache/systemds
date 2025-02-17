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
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.hops.fedplanner.FTypes.AlignType;
import org.apache.sysds.hops.fedplanner.FTypes.FType;
import org.apache.sysds.lops.MapMult;
import org.apache.sysds.lops.PMMJ;
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
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.spark.AggregateBinarySPInstruction;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.AggregateBinaryOperator;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;

public class MMFEDInstruction extends BinaryFEDInstruction
{
	private MMFEDInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out, String opcode, String istr) {
		super(FEDType.MAPMM, op, in1, in2, out, opcode, istr);
	}

	public static MMFEDInstruction parseInstruction(AggregateBinarySPInstruction instr) {
		return new MMFEDInstruction(instr.getOperator(), instr.input1, instr.input2, instr.output, instr.getOpcode(),
			instr.getInstructionString());
	}

	public static MMFEDInstruction parseInstruction( String str ) {
		String parts[] = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];

		if(!ArrayUtils.contains(new String[] {MapMult.OPCODE, PMMJ.OPCODE, "cpmm", "rmm"}, opcode))
			throw new DMLRuntimeException("MapmmSPInstruction.parseInstruction():: Unknown opcode " + opcode);

		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand in2 = new CPOperand(parts[2]);
		CPOperand out = new CPOperand(parts[3]);

		AggregateBinaryOperator aggbin = InstructionUtils.getMatMultOperator(1);
		return new MMFEDInstruction(aggbin, in1, in2, out, opcode, str);
	}

	public void processInstruction(ExecutionContext ec) {
		MatrixLineagePair mo1 = ec.getMatrixLineagePair(input1);
		MatrixLineagePair mo2 = ec.getMatrixLineagePair(input2);

		long id = FederationUtils.getNextFedDataID();
		FederatedRequest frEmpty = new FederatedRequest(FederatedRequest.RequestType.PUT_VAR,
			id, new MatrixCharacteristics(-1, -1), DataType.MATRIX);

		//TODO cleanup unnecessary redundancy
		//#1 federated matrix-vector multiplication
		if(mo1.isFederated(FType.COL) && mo2.isFederated(FType.ROW)
			&& mo1.getFedMapping().isAligned(mo2.getFedMapping(), AlignType.COL_T) ) {
			FederatedRequest fr1 = FederationUtils.callInstruction(instString, output, id,
				new CPOperand[]{input1, input2},
				new long[]{mo1.getFedMapping().getID(), mo2.getFedMapping().getID()}, ExecType.SPARK, false);

			if ( _fedOut.isForcedFederated() ){
				mo1.getFedMapping().execute(getTID(), frEmpty, fr1);
				setPartialOutput(mo1.getFedMapping(), mo1.getMO(), mo2.getMO(), fr1.getID(), ec);
			}
			else {
				aggregateLocally(mo1.getFedMapping(), true, ec, frEmpty, fr1);
			}
		}
		else if(mo1.isFederated(FType.ROW) || mo1.isFederated(FType.PART)) { // MV + MM
			//construct commands: broadcast rhs, fed mv, retrieve results
			FederatedRequest fr1 = mo1.getFedMapping().broadcast(mo2);
			FederatedRequest fr2 = FederationUtils.callInstruction(instString, output, id,
				new CPOperand[]{input1, input2},
				new long[]{mo1.getFedMapping().getID(), fr1.getID()}, ExecType.SPARK, false);

			boolean isVector = (mo2.getNumColumns() == 1);
			boolean isPartOut = mo1.isFederated(FType.PART) || // MV and MM
				(!isVector && mo2.isFederated(FType.PART)); // only MM
			if(isPartOut && _fedOut.isForcedFederated()) {
				mo1.getFedMapping().execute(getTID(), true, frEmpty, fr1, fr2);
				setPartialOutput(mo1.getFedMapping(), mo1.getMO(), mo2.getMO(), fr2.getID(), ec);
			}
			else if((_fedOut.isForcedFederated() || (!isVector && !_fedOut.isForcedLocal()))
				&& !isPartOut) { // not creating federated output in the MV case for reasons of performance
				mo1.getFedMapping().execute(getTID(), true, frEmpty, fr1, fr2);
				setOutputFedMapping(mo1.getFedMapping(), mo1.getMO(), mo2.getMO(), fr2.getID(), ec);
			}
			else {
				aggregateLocally(mo1.getFedMapping(), mo1.isFederated(FType.PART), ec, frEmpty, fr1, fr2);
			}
		}
		//#2 vector - federated matrix multiplication
		else if (mo2.isFederated(FType.ROW)) {// VM + MM
			//construct commands: broadcast rhs, fed mv, retrieve results
			FederatedRequest[] fr1 = mo2.getFedMapping().broadcastSliced(mo1, true);
			FederatedRequest fr2 = FederationUtils.callInstruction(instString, output, id,
				new CPOperand[]{input1, input2},
				new long[]{fr1[0].getID(), mo2.getFedMapping().getID()}, ExecType.SPARK, false);
			if ( _fedOut.isForcedFederated() ){
				// Partial aggregates (set fedmapping to the partial aggs)
				mo2.getFedMapping().execute(getTID(), true, fr1, frEmpty, fr2);
				setPartialOutput(mo2.getFedMapping(), mo1.getMO(), mo2.getMO(), fr2.getID(), ec);
			}
			else {
				aggregateLocally(mo2.getFedMapping(), true, ec, fr1, frEmpty, fr2);
			}
		}
		//#3 col-federated matrix vector multiplication
		else if (mo1.isFederated(FType.COL)) {// VM + MM
			//construct commands: broadcast rhs, fed mv, retrieve results
			FederatedRequest[] fr1 = mo1.getFedMapping().broadcastSliced(mo2, true);
			FederatedRequest fr2 = FederationUtils.callInstruction(instString, output, id,
				new CPOperand[]{input1, input2},
				new long[]{mo1.getFedMapping().getID(), fr1[0].getID()}, ExecType.SPARK, false);
			if ( _fedOut.isForcedFederated() ){
				// Partial aggregates (set fedmapping to the partial aggs)
				mo1.getFedMapping().execute(getTID(), true, fr1, frEmpty, fr2);
				setPartialOutput(mo1.getFedMapping(), mo1.getMO(), mo2.getMO(), fr2.getID(), ec);
			}
			else {
				aggregateLocally(mo1.getFedMapping(), true, ec, fr1, frEmpty, fr2);
			}
		}
		else { //other combinations
			throw new DMLRuntimeException("Federated AggregateBinary not supported with the "
				+ "following federated objects: "+mo1.isFederated()+":"+mo1.getFedMapping()
				+" "+mo2.isFederated()+":"+mo2.getFedMapping());
		}
	}

	/**
	 * Sets the output with a federated mapping of overlapping partial aggregates.
	 * @param federationMap federated map from which the federated metadata is retrieved
	 * @param mo1 matrix object with number of rows used to set the number of rows of the output
	 * @param mo2 matrix object with number of columns used to set the number of columns of the output
	 * @param outputID ID of the output
	 * @param ec execution context
	 */
	private void setPartialOutput(FederationMap federationMap, MatrixObject mo1, MatrixObject mo2,
		long outputID, ExecutionContext ec){
		MatrixObject out = ec.getMatrixObject(output);
		out.getDataCharacteristics().set(mo1.getNumRows(), mo2.getNumColumns(), mo1.getBlocksize());
		FederationMap outputFedMap = federationMap
			.copyWithNewIDAndRange(mo1.getNumRows(), mo2.getNumColumns(), outputID);
		out.setFedMapping(outputFedMap);
	}

	/**
	 * Sets the output with a federated map copied from federationMap input given.
	 * @param federationMap federation map to be set in output
	 * @param mo1 matrix object with number of rows used to set the number of rows of the output
	 * @param mo2 matrix object with number of columns used to set the number of columns of the output
	 * @param outputID ID of the output
	 * @param ec execution context
	 */
	private void setOutputFedMapping(FederationMap federationMap, MatrixObject mo1, MatrixObject mo2,
		long outputID, ExecutionContext ec){
		MatrixObject out = ec.getMatrixObject(output);
		out.getDataCharacteristics().set(mo1.getNumRows(), mo2.getNumColumns(), mo1.getBlocksize());
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
	private void aggregateLocally(FederationMap fedMap, boolean aggAdd,
		ExecutionContext ec, FederatedRequest[] frSliced, FederatedRequest... fr) {
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
}
