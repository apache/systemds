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

import org.apache.sysds.runtime.codegen.SpoofCellwise;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.sysds.runtime.codegen.CodegenUtils;
import org.apache.sysds.runtime.codegen.SpoofOperator;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest.RequestType;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse;
import org.apache.sysds.runtime.controlprogram.federated.FederationMap;
import org.apache.sysds.runtime.controlprogram.federated.FederationMap.FType;
import org.apache.sysds.runtime.controlprogram.federated.FederationUtils;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;

import java.util.ArrayList;
import java.util.Collections;
import java.util.concurrent.Future;
import java.util.stream.Stream;

public class SpoofFEDInstruction extends FEDInstruction
{
	private final Class<?> _class;
	private final byte[] _class_bytes;
	private final SpoofOperator _op;
	private final CPOperand[] _inputs;
	private final CPOperand _output;

	private SpoofFEDInstruction(SpoofOperator op, Class<?> cla, byte[] class_bytes, CPOperand[] in,
			CPOperand out, String opcode, String inst_str)
	{
		super(FEDInstruction.FEDType.SpoofFused, opcode, inst_str);
		_class = cla;
		_class_bytes = class_bytes;
		_op = op;
		_inputs = in;
		_output = out;
	}

	public static SpoofFEDInstruction parseInstruction(String str)
	{
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);

		// ArrayList<CPOperand> inputList = new ArrayList<CPOperand>();
		CPOperand[] inputCpo = new CPOperand[parts.length - 3 - 2];
		Class<?> cla = CodegenUtils.getClass(parts[2]);
		SpoofOperator op = CodegenUtils.createInstance(cla);
		byte[] classBytes = CodegenUtils.getClassData(parts[2]);
		String opcode = parts[0] + op.getSpoofType();

		for(int counter = 3; counter < parts.length - 2; counter++) {
			// inputList.add(new CPOperand(parts[counter]));
			inputCpo[counter - 3] = new CPOperand(parts[counter]);
		}
		CPOperand out = new CPOperand(parts[parts.length - 2]);

		// return new SpoofFEDInstruction(op, cls, classBytes, inputList.toArray(new CPOperand[0]), out, opcode, str);
		return new SpoofFEDInstruction(op, cla, classBytes, inputCpo, out, opcode, str);
	}

	@Override
	public void processInstruction(ExecutionContext ec)
	{
		ArrayList<CPOperand> inCpoMat = new ArrayList<CPOperand>();
		ArrayList<CPOperand> inCpoScal = new ArrayList<CPOperand>();
		ArrayList<MatrixObject> inMo = new ArrayList<MatrixObject>();
		ArrayList<ScalarObject> inSo = new ArrayList<ScalarObject>();
		MatrixObject fedMo = null;
		FederationMap fedMap = null;
		for(CPOperand cpo : _inputs) {
			Data tmpData = ec.getVariable(cpo);
			if(tmpData instanceof MatrixObject) {
				MatrixObject tmp = (MatrixObject) tmpData;
				if(tmp.isFederated()) {
					inCpoMat.add(0, cpo); // insert federated CPO at the beginning
					fedMo = tmp;
					fedMap = tmp.getFedMapping();
				}
				else {
					inCpoMat.add(cpo);
					inMo.add(tmp);
				}
			}
			else if(tmpData instanceof ScalarObject) {
				ScalarObject tmp = (ScalarObject) tmpData;
				inCpoScal.add(cpo);
				inSo.add(tmp);
			}
		}

		ArrayList<FederatedRequest> frBroadcast = new ArrayList<FederatedRequest>();
		ArrayList<FederatedRequest[]> frBroadcastSliced = new ArrayList<FederatedRequest[]>();
		long[] frIds = new long[1 + inMo.size() + inSo.size()];
		int index = 0;
		frIds[index++] = fedMap.getID(); // insert federation map id at the beginning
		for(MatrixObject mo : inMo) {
			if((fedMo.isFederated(FType.ROW) && mo.getNumRows() > 1 && mo.getNumColumns() == 1)
				|| (fedMo.isFederated(FType.COL) && mo.getNumRows() == 1 && mo.getNumColumns() > 1)) {
				FederatedRequest[] tmpFr = fedMap.broadcastSliced(mo, false);
				frIds[index++] = tmpFr[0].getID();
				frBroadcastSliced.add(tmpFr);
			}
			else {
				FederatedRequest tmpFr = fedMap.broadcast(mo);
				frIds[index++] = tmpFr.getID();
				frBroadcast.add(tmpFr);
			}
		}
		for(ScalarObject so : inSo) {
			FederatedRequest tmpFr = fedMap.broadcast(so);
			frIds[index++] = tmpFr.getID();
			frBroadcast.add(tmpFr);
		}

		// change the is_literal flag from true to false because when broadcasted it is no literal anymore
		instString = instString.replace("true", "false");

		// ArrayList<CPOperand> inCpo = new ArrayList<>(inCpoMat);
		CPOperand[] inCpo = ArrayUtils.addAll(inCpoMat.toArray(new CPOperand[0]), inCpoScal.toArray(new CPOperand[0]));
		// inCpo.addAll(inCpoScal);

		// FederatedRequest frCompute = FederationUtils.callInstruction(instString, _output,
		// 	Stream.of(inCpoMat.toArray(new CPOperand[0]), inCpoScal.toArray(new CPOperand[0])).flatMap(Stream::of).toArray(CPOperand[]::new), frIds);
		// FederatedRequest frCompute = FederationUtils.callInstruction(instString, _output,
		// 	inCpo.toArray(new CPOperand[0]), frIds);
		FederatedRequest frCompute = FederationUtils.callInstruction(instString, _output,
			inCpo, frIds);

		// get partial results from federated workers
		FederatedRequest frGet = new FederatedRequest(RequestType.GET_VAR, frCompute.getID());

		// assert false: "SpoofFEDInstruction.java:138 - cpos: " + Stream.of(inCpoMat.toArray(new CPOperand[0]), inCpoScal.toArray(new CPOperand[0])).flatMap(Stream::of).toArray(CPOperand[]::new).length;

		ArrayList<FederatedRequest> frCleanup = new ArrayList<FederatedRequest>();
		frCleanup.add(fedMap.cleanup(getTID(), frCompute.getID()));
		for(FederatedRequest fr : frBroadcast) {
			frCleanup.add(fedMap.cleanup(getTID(), fr.getID()));
		}
		for(FederatedRequest[] fr : frBroadcastSliced) {
			frCleanup.add(fedMap.cleanup(getTID(), fr[0].getID()));
		}

		// *************************************************************************
		// *************************************************************************
		// *************************************************************************
		// *************************************************************************
		// ArrayList<FederatedRequest> frAll = new ArrayList<>(frBroadcast);
		// frAll.add(frCompute);
		// frAll.add(frGet);
		// frAll.addAll(frCleanup);
		FederatedRequest[] frAll = ArrayUtils.addAll(ArrayUtils.addAll(frBroadcast.toArray(new FederatedRequest[0]), frCompute, frGet), frCleanup.toArray(new FederatedRequest[0]));
		// Future<FederatedResponse>[] response = fedMap.executeMultipleSlices(
		// 	getTID(), true, frBroadcastSliced.toArray(new FederatedRequest[0][]),
		// 	frAll.toArray(new FederatedRequest[0]));
		Future<FederatedResponse>[] response = fedMap.executeMultipleSlices(
			getTID(), true, frBroadcastSliced.toArray(new FederatedRequest[0][]),
			frAll);

		if(_output.isScalar() && ((SpoofCellwise)_op).getAggOp() == SpoofCellwise.AggOp.SUM) {
			//aggregate partial results from federated responses
			AggregateUnaryOperator aop = InstructionUtils.parseBasicAggregateUnaryOperator("uak+");
			ec.setVariable(_output.getName(), FederationUtils.aggScalar(aop, response));
		}
		else if(fedMo.isFederated(FType.ROW)) {
			// bind partial results from federated responses
			ec.setMatrixOutput(_output.getName(), FederationUtils.bind(response, false));
		}
		else if(fedMo.isFederated(FType.COL)) {
			// bind partial results from federated responses
			ec.setMatrixOutput(_output.getName(), FederationUtils.bind(response, true));
		}

	}

}

