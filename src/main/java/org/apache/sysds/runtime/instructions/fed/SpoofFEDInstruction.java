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
import org.apache.sysds.runtime.codegen.SpoofRowwise;
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

	private final SpoofType _spoof_type;

	private SpoofFEDInstruction(SpoofOperator op, CPOperand[] in,
			CPOperand out, String opcode, String instStr, SpoofType spoofType)
	{
		super(FEDInstruction.FEDType.SpoofFused, opcode, instStr);
		_op = op;
		_inputs = in;
		_output = out;
		_spoof_type = spoofType;
	}
	
	public enum SpoofType {
		CELLWISE,
		ROWWISE,
		UNKNOWN
	}

	public static SpoofFEDInstruction parseInstruction(String str)
	{
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);

		CPOperand[] inputCpo = new CPOperand[parts.length - 3 - 2];
		Class<?> cla = CodegenUtils.getClass(parts[2]);
		SpoofOperator op = CodegenUtils.createInstance(cla);
		String opcode = parts[0] + op.getSpoofType();

		for(int counter = 3; counter < parts.length - 2; counter++)
			inputCpo[counter - 3] = new CPOperand(parts[counter]);
		CPOperand out = new CPOperand(parts[parts.length - 2]);

		SpoofType spoofType = SpoofType.UNKNOWN;
		if(op.getClass().getSuperclass() == SpoofCellwise.class)
			spoofType = SpoofType.CELLWISE;
		else if(op.getClass().getSuperclass() == SpoofRowwise.class)
			spoofType = SpoofType.ROWWISE;

		return new SpoofFEDInstruction(op, inputCpo, out, opcode, str, spoofType);
	}

	@Override
	public void processInstruction(ExecutionContext ec)
	{
		ArrayList<CPOperand> inCpoMat = new ArrayList<>();
		ArrayList<CPOperand> inCpoScal = new ArrayList<>();
		ArrayList<MatrixObject> inMo = new ArrayList<>();
		ArrayList<ScalarObject> inSo = new ArrayList<>();
		FederationMap fedMap = null;
		for(CPOperand cpo : _inputs) {
			Data tmpData = ec.getVariable(cpo);
			if(tmpData instanceof MatrixObject) {
				MatrixObject tmp = (MatrixObject) tmpData;
				if(fedMap == null & tmp.isFederated()) { //take first
					inCpoMat.add(0, cpo); // insert federated CPO at the beginning
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

		ArrayList<FederatedRequest> frBroadcast = new ArrayList<>();
		ArrayList<FederatedRequest[]> frBroadcastSliced = new ArrayList<>();
		long[] frIds = new long[1 + inMo.size() + inSo.size()];
		int index = 0;
		frIds[index++] = fedMap.getID(); // insert federation map id at the beginning
		for(MatrixObject mo : inMo) {
			if(needsBroadcastSliced(fedMap, mo.getNumRows(), mo.getNumColumns())) {
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
		FederatedRequest frCompute = FederationUtils.callInstruction(instString, _output, inCpo, frIds);

		// get partial results from federated workers
		FederatedRequest frGet = new FederatedRequest(RequestType.GET_VAR, frCompute.getID());

		ArrayList<FederatedRequest> frCleanup = new ArrayList<>();
		frCleanup.add(fedMap.cleanup(getTID(), frCompute.getID()));
		for(FederatedRequest fr : frBroadcast)
			frCleanup.add(fedMap.cleanup(getTID(), fr.getID()));
		for(FederatedRequest[] fr : frBroadcastSliced)
			frCleanup.add(fedMap.cleanup(getTID(), fr[0].getID()));

		FederatedRequest[] frAll = ArrayUtils.addAll(ArrayUtils.addAll(
			frBroadcast.toArray(new FederatedRequest[0]), frCompute, frGet),
			frCleanup.toArray(new FederatedRequest[0]));
		Future<FederatedResponse>[] response = fedMap.executeMultipleSlices(
			getTID(), true, frBroadcastSliced.toArray(new FederatedRequest[0][]), frAll);

		if(_spoof_type == SpoofType.CELLWISE) {
			setOutputCellwise(ec, response, fedMap);
		}
		else if(_spoof_type == SpoofType.ROWWISE) {
			setOutputRowwise(ec, response, fedMap);
		}
		else {
			throw new DMLRuntimeException("Federated code generation only supported for cellwise and rowwise templates.");
		}
	}

	private boolean needsBroadcastSliced(FederationMap fedMap, long rowNum, long colNum)
	{
		if(fedMap.getType() == FType.ROW) {
			if(_spoof_type == SpoofType.CELLWISE)
				return (rowNum > 1 && (colNum == 1 || colNum == fedMap.getSize()))
					|| (colNum > 1 && rowNum == fedMap.getSize());
			else
				return (rowNum > 1 && colNum == fedMap.getSize())
					|| (colNum > 1 && rowNum == fedMap.getSize());
		}
		else if(fedMap.getType() == FType.COL) {
			return ((rowNum == 1 || rowNum == fedMap.getSize()) && colNum > 1)
				|| (rowNum > 1 && colNum == fedMap.getSize());
		}
		return false;
	}

	private void setOutputCellwise(ExecutionContext ec, Future<FederatedResponse>[] response, FederationMap fedMap)
	{
		FType fedType = fedMap.getType();
		if(((SpoofCellwise)_op).getCellType() == SpoofCellwise.CellType.FULL_AGG) { // full aggregation
			if(((SpoofCellwise)_op).getAggOp() == SpoofCellwise.AggOp.SUM
				|| ((SpoofCellwise)_op).getAggOp() == SpoofCellwise.AggOp.SUM_SQ) {
				// aggregate partial results from federated responses as sum
				AggregateUnaryOperator aop = InstructionUtils.parseBasicAggregateUnaryOperator("uak+");
				ec.setVariable(_output.getName(), FederationUtils.aggScalar(aop, response));
			}
			else if(((SpoofCellwise)_op).getAggOp() == SpoofCellwise.AggOp.MIN) {
				// aggregate partial results from federated responses as min
				AggregateUnaryOperator aop = InstructionUtils.parseBasicAggregateUnaryOperator("uamin");
				ec.setVariable(_output.getName(), FederationUtils.aggScalar(aop, response));
			}
			else if(((SpoofCellwise)_op).getAggOp() == SpoofCellwise.AggOp.MAX) {
				// aggregate partial results from federated responses as max
				AggregateUnaryOperator aop = InstructionUtils.parseBasicAggregateUnaryOperator("uamax");
				ec.setVariable(_output.getName(), FederationUtils.aggScalar(aop, response));
			}
			else {
				throw new DMLRuntimeException("Aggregation type for federated spoof instructions not supported yet.");
			}
		}
		else if(((SpoofCellwise)_op).getCellType() == SpoofCellwise.CellType.ROW_AGG) { // row aggregation
			if(fedType == FType.ROW) {
				// bind partial results from federated responses
				ec.setMatrixOutput(_output.getName(), FederationUtils.bind(response, false));
			}
			else if(fedType == FType.COL) {
				if(((SpoofCellwise)_op).getAggOp() == SpoofCellwise.AggOp.SUM
				|| ((SpoofCellwise)_op).getAggOp() == SpoofCellwise.AggOp.SUM_SQ) {
					AggregateUnaryOperator aop = InstructionUtils.parseBasicAggregateUnaryOperator("uark+");
					ec.setMatrixOutput(_output.getName(), FederationUtils.aggMatrix(aop, response, fedMap));
				}
				else if(((SpoofCellwise)_op).getAggOp() == SpoofCellwise.AggOp.MIN) {
					AggregateUnaryOperator aop = InstructionUtils.parseBasicAggregateUnaryOperator("uarmin");
					ec.setMatrixOutput(_output.getName(), FederationUtils.aggMatrix(aop, response, fedMap));
				}
				else if(((SpoofCellwise)_op).getAggOp() == SpoofCellwise.AggOp.MAX) {
					AggregateUnaryOperator aop = InstructionUtils.parseBasicAggregateUnaryOperator("uarmax");
					ec.setMatrixOutput(_output.getName(), FederationUtils.aggMatrix(aop, response, fedMap));
				}
			}
			else {
				throw new DMLRuntimeException("Aggregation type for federated spoof instructions not supported yet.");
			}
		}
		else if(((SpoofCellwise)_op).getCellType() == SpoofCellwise.CellType.COL_AGG) { // col aggregation
			if(fedType == FType.ROW) {
				if(((SpoofCellwise)_op).getAggOp() == SpoofCellwise.AggOp.SUM
					|| ((SpoofCellwise)_op).getAggOp() == SpoofCellwise.AggOp.SUM_SQ) {
					AggregateUnaryOperator aop = InstructionUtils.parseBasicAggregateUnaryOperator("uack+");
					ec.setMatrixOutput(_output.getName(), FederationUtils.aggMatrix(aop, response, fedMap));
				}
				else if(((SpoofCellwise)_op).getAggOp() == SpoofCellwise.AggOp.MIN) {
					AggregateUnaryOperator aop = InstructionUtils.parseBasicAggregateUnaryOperator("uacmin");
					ec.setMatrixOutput(_output.getName(), FederationUtils.aggMatrix(aop, response, fedMap));
				}
				else if(((SpoofCellwise)_op).getAggOp() == SpoofCellwise.AggOp.MAX) {
					AggregateUnaryOperator aop = InstructionUtils.parseBasicAggregateUnaryOperator("uacmax");
					ec.setMatrixOutput(_output.getName(), FederationUtils.aggMatrix(aop, response, fedMap));
				}
			}
			else if(fedType == FType.COL) {
				// bind partial results from federated responses
				ec.setMatrixOutput(_output.getName(), FederationUtils.bind(response, true));
			}
			else {
				throw new DMLRuntimeException("Aggregation type for federated spoof instructions not supported yet.");
			}
		}
		else if(((SpoofCellwise)_op).getCellType() == SpoofCellwise.CellType.NO_AGG) { // no aggregation
			if(fedType == FType.ROW) {
				// bind partial results from federated responses
				ec.setMatrixOutput(_output.getName(), FederationUtils.bind(response, false));
			}
			else if(fedType == FType.COL) {
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

	private void setOutputRowwise(ExecutionContext ec, Future<FederatedResponse>[] response, FederationMap fedMap)
	{
		if(((SpoofRowwise)_op).getRowType() == SpoofRowwise.RowType.FULL_AGG) { // full aggregation
			// aggregate partial results from federated responses as sum
			AggregateUnaryOperator aop = InstructionUtils.parseBasicAggregateUnaryOperator("uak+");
			ec.setVariable(_output.getName(), FederationUtils.aggScalar(aop, response));
		}
		else if(((SpoofRowwise)_op).getRowType() == SpoofRowwise.RowType.ROW_AGG) { // row aggregation
			// aggregate partial results from federated responses as rowSum
			AggregateUnaryOperator aop = InstructionUtils.parseBasicAggregateUnaryOperator("uark+");
			ec.setMatrixOutput(_output.getName(), FederationUtils.aggMatrix(aop, response, fedMap));
		}
		else if(((SpoofRowwise)_op).getRowType() == SpoofRowwise.RowType.COL_AGG
		|| ((SpoofRowwise)_op).getRowType() == SpoofRowwise.RowType.COL_AGG_T
		|| ((SpoofRowwise)_op).getRowType() == SpoofRowwise.RowType.COL_AGG_B1
		|| ((SpoofRowwise)_op).getRowType() == SpoofRowwise.RowType.COL_AGG_B1_T
		|| ((SpoofRowwise)_op).getRowType() == SpoofRowwise.RowType.COL_AGG_B1R
		|| ((SpoofRowwise)_op).getRowType() == SpoofRowwise.RowType.COL_AGG_CONST) { // col aggregation
			// aggregate partial results from federated responses as colSum
			AggregateUnaryOperator aop = InstructionUtils.parseBasicAggregateUnaryOperator("uack+");
			ec.setMatrixOutput(_output.getName(), FederationUtils.aggMatrix(aop, response, fedMap));
		}
		else if(((SpoofRowwise)_op).getRowType() == SpoofRowwise.RowType.NO_AGG
			|| ((SpoofRowwise)_op).getRowType() == SpoofRowwise.RowType.NO_AGG_B1
			|| ((SpoofRowwise)_op).getRowType() == SpoofRowwise.RowType.NO_AGG_CONST) { // no aggregation
			if(fedMap.getType() == FType.ROW) {
				// bind partial results from federated responses
				ec.setMatrixOutput(_output.getName(), FederationUtils.bind(response, false));
			}
			else {
				throw new DMLRuntimeException("Only row partitioned federated matrices supported yet.");
			}
		}
		else {
			throw new DMLRuntimeException("AggregationType not supported yet.");
		}
	}

}
