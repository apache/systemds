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

import org.apache.commons.lang3.ArrayUtils;
import org.apache.sysds.runtime.codegen.CodegenUtils;
import org.apache.sysds.runtime.codegen.SpoofCellwise;
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
import java.util.concurrent.Future;

public class SpoofFEDInstruction extends FEDInstruction
{
	private final SpoofOperator _op;
	private final CPOperand[] _inputs;
	private final CPOperand _output;

	private SpoofFEDInstruction(SpoofOperator op, CPOperand[] in,
			CPOperand out, String opcode, String inst_str)
	{
		super(FEDInstruction.FEDType.SpoofFused, opcode, inst_str);
		_op = op;
		_inputs = in;
		_output = out;
	}

	public static SpoofFEDInstruction parseInstruction(String str)
	{
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);

		CPOperand[] inputCpo = new CPOperand[parts.length - 3 - 2];
		Class<?> cla = CodegenUtils.getClass(parts[2]);
		SpoofOperator op = CodegenUtils.createInstance(cla);
		String opcode = parts[0] + op.getSpoofType();

		for(int counter = 3; counter < parts.length - 2; counter++) {
			inputCpo[counter - 3] = new CPOperand(parts[counter]);
		}
		CPOperand out = new CPOperand(parts[parts.length - 2]);

		return new SpoofFEDInstruction(op, inputCpo, out, opcode, str);
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
			if((fedMo.isFederated(FType.ROW) && mo.getNumRows() > 1 && (mo.getNumColumns() == 1 || mo.getNumColumns() == fedMap.getSize()))
				|| (fedMo.isFederated(FType.ROW) && mo.getNumColumns() > 1 && mo.getNumRows() == fedMap.getSize())
				|| (fedMo.isFederated(FType.COL) && (mo.getNumRows() == 1 || mo.getNumRows() == fedMap.getSize()) && mo.getNumColumns() > 1)
				|| (fedMo.isFederated(FType.COL) && mo.getNumRows() > 1 && mo.getNumColumns() == fedMap.getSize())) {
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

		// change the is_literal flag from true to false because when broadcasted it is not a literal anymore
		instString = instString.replace("true", "false");

		CPOperand[] inCpo = ArrayUtils.addAll(inCpoMat.toArray(new CPOperand[0]), inCpoScal.toArray(new CPOperand[0]));

		FederatedRequest frCompute = FederationUtils.callInstruction(instString, _output,
			inCpo, frIds);

		// get partial results from federated workers
		FederatedRequest frGet = new FederatedRequest(RequestType.GET_VAR, frCompute.getID());

		ArrayList<FederatedRequest> frCleanup = new ArrayList<FederatedRequest>();
		frCleanup.add(fedMap.cleanup(getTID(), frCompute.getID()));
		for(FederatedRequest fr : frBroadcast) {
			frCleanup.add(fedMap.cleanup(getTID(), fr.getID()));
		}
		for(FederatedRequest[] fr : frBroadcastSliced) {
			frCleanup.add(fedMap.cleanup(getTID(), fr[0].getID()));
		}

		FederatedRequest[] frAll = ArrayUtils.addAll(ArrayUtils.addAll(frBroadcast.toArray(new FederatedRequest[0]), frCompute, frGet), frCleanup.toArray(new FederatedRequest[0]));
		Future<FederatedResponse>[] response = fedMap.executeMultipleSlices(
			getTID(), true, frBroadcastSliced.toArray(new FederatedRequest[0][]),
			frAll);

		if(((SpoofCellwise)_op).getCellType() == SpoofCellwise.CellType.FULL_AGG) { // full aggregation
			if(((SpoofCellwise)_op).getAggOp() == SpoofCellwise.AggOp.SUM
				|| ((SpoofCellwise)_op).getAggOp() == SpoofCellwise.AggOp.SUM_SQ) {
				//aggregate partial results from federated responses as sum
				AggregateUnaryOperator aop = InstructionUtils.parseBasicAggregateUnaryOperator("uak+");
				ec.setVariable(_output.getName(), FederationUtils.aggScalar(aop, response));
			}
			else if(((SpoofCellwise)_op).getAggOp() == SpoofCellwise.AggOp.MIN) {
				//aggregate partial results from federated responses as min
				AggregateUnaryOperator aop = InstructionUtils.parseBasicAggregateUnaryOperator("uamin");
				ec.setVariable(_output.getName(), FederationUtils.aggScalar(aop, response));
			}
			else if(((SpoofCellwise)_op).getAggOp() == SpoofCellwise.AggOp.MAX) {
				//aggregate partial results from federated responses as max
				AggregateUnaryOperator aop = InstructionUtils.parseBasicAggregateUnaryOperator("uamax");
				ec.setVariable(_output.getName(), FederationUtils.aggScalar(aop, response));
			}
			else {
				throw new DMLRuntimeException("Aggregation type for federated spoof instructions not supported yet.");
			}
		}
		else if(((SpoofCellwise)_op).getCellType() == SpoofCellwise.CellType.ROW_AGG) { // row aggregation
			if(fedMo.isFederated(FType.ROW)) {
				// bind partial results from federated responses
				ec.setMatrixOutput(_output.getName(), FederationUtils.bind(response, false));
			}
			else if(fedMo.isFederated(FType.COL)) {
				if(((SpoofCellwise)_op).getAggOp() == SpoofCellwise.AggOp.SUM
				|| ((SpoofCellwise)_op).getAggOp() == SpoofCellwise.AggOp.SUM_SQ) {
					//aggregate partial results from federated responses as rowSum
					AggregateUnaryOperator aop = InstructionUtils.parseBasicAggregateUnaryOperator("uark+");
					ec.setMatrixOutput(_output.getName(), FederationUtils.aggMatrix(aop, response, fedMap));
				}
				else if(((SpoofCellwise)_op).getAggOp() == SpoofCellwise.AggOp.MIN) {
					//aggregate partial results from federated responses as rowMin
					AggregateUnaryOperator aop = InstructionUtils.parseBasicAggregateUnaryOperator("uarmin");
					ec.setMatrixOutput(_output.getName(), FederationUtils.aggMatrix(aop, response, fedMap));
				}
				else if(((SpoofCellwise)_op).getAggOp() == SpoofCellwise.AggOp.MAX) {
					//aggregate partial results from federated responses as rowMax
					AggregateUnaryOperator aop = InstructionUtils.parseBasicAggregateUnaryOperator("uarmax");
					ec.setMatrixOutput(_output.getName(), FederationUtils.aggMatrix(aop, response, fedMap));
				}
			}
			else {
				throw new DMLRuntimeException("Aggregation type for federated spoof instructions not supported yet.");
			}
		}
		else if(((SpoofCellwise)_op).getCellType() == SpoofCellwise.CellType.COL_AGG) { // col aggregation
			if(fedMo.isFederated(FType.ROW)) {
				if(((SpoofCellwise)_op).getAggOp() == SpoofCellwise.AggOp.SUM
					|| ((SpoofCellwise)_op).getAggOp() == SpoofCellwise.AggOp.SUM_SQ) {
					//aggregate partial results from federated responses as colSum
					AggregateUnaryOperator aop = InstructionUtils.parseBasicAggregateUnaryOperator("uack+");
					ec.setMatrixOutput(_output.getName(), FederationUtils.aggMatrix(aop, response, fedMap));
				}
				else if(((SpoofCellwise)_op).getAggOp() == SpoofCellwise.AggOp.MIN) {
					//aggregate partial results from federated responses as colMin
					AggregateUnaryOperator aop = InstructionUtils.parseBasicAggregateUnaryOperator("uacmin");
					ec.setMatrixOutput(_output.getName(), FederationUtils.aggMatrix(aop, response, fedMap));
				}
				else if(((SpoofCellwise)_op).getAggOp() == SpoofCellwise.AggOp.MAX) {
					//aggregate partial results from federated responses as colMax
					AggregateUnaryOperator aop = InstructionUtils.parseBasicAggregateUnaryOperator("uacmax");
					ec.setMatrixOutput(_output.getName(), FederationUtils.aggMatrix(aop, response, fedMap));
				}
			}
			else if(fedMo.isFederated(FType.COL)) {
				// bind partial results from federated responses
				ec.setMatrixOutput(_output.getName(), FederationUtils.bind(response, true));
			}
			else {
				throw new DMLRuntimeException("Aggregation type for federated spoof instructions not supported yet.");
			}
		}
		else if(((SpoofCellwise)_op).getCellType() == SpoofCellwise.CellType.NO_AGG) { // no aggregation
			if(fedMo.isFederated(FType.ROW)) {
				// bind partial results from federated responses
				ec.setMatrixOutput(_output.getName(), FederationUtils.bind(response, false));
			}
			else if(fedMo.isFederated(FType.COL)) {
				// bind partial results from federated responses
				ec.setMatrixOutput(_output.getName(), FederationUtils.bind(response, true));
			}
			else {
				throw new DMLRuntimeException("Only row partitioned or column partitioned federated matrices supported yet.");
			}
		}
		else {
			throw new DMLRuntimeException("Aggregation type not supported yet.");
		}
	}

}

