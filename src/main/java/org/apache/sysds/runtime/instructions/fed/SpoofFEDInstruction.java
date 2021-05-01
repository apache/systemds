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
import org.apache.sysds.runtime.codegen.SpoofCellwise.AggOp;
import org.apache.sysds.runtime.codegen.SpoofCellwise.CellType;
import org.apache.sysds.runtime.codegen.SpoofMultiAggregate;
import org.apache.sysds.runtime.codegen.SpoofOperator;
import org.apache.sysds.runtime.codegen.SpoofOuterProduct;
import org.apache.sysds.runtime.codegen.SpoofOuterProduct.OutProdType;
import org.apache.sysds.runtime.codegen.SpoofRowwise;
import org.apache.sysds.runtime.codegen.SpoofRowwise.RowType;
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
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;

import java.util.ArrayList;
import java.util.concurrent.Future;

public class SpoofFEDInstruction extends FEDInstruction
{
	private final SpoofOperator _op;
	private final CPOperand[] _inputs;
	private final CPOperand _output;

	private SpoofFEDInstruction(SpoofOperator op, CPOperand[] in,
		CPOperand out, String opcode, String instStr)
	{
		super(FEDInstruction.FEDType.SpoofFused, opcode, instStr);
		_op = op;
		_inputs = in;
		_output = out;
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

		return new SpoofFEDInstruction(op, inputCpo, out, opcode, str);
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

		if(_op.getClass().getSuperclass() == SpoofCellwise.class)
			setOutputCellwise(ec, response, fedMap);
		else if(_op.getClass().getSuperclass() == SpoofRowwise.class)
			setOutputRowwise(ec, response, fedMap);
		else if(_op.getClass().getSuperclass() == SpoofMultiAggregate.class)
			setOutputMultiAgg(ec, response, fedMap);
		else if(_op.getClass().getSuperclass() == SpoofOuterProduct.class)
			setOutputOuterProduct(ec, response, fedMap);
		else
			throw new DMLRuntimeException("Federated code generation only supported for cellwise, rowwise, and multiaggregate templates.");
	}

	private static boolean needsBroadcastSliced(FederationMap fedMap, long rowNum, long colNum) {
		if(rowNum == fedMap.getMaxIndexInRange(0) && colNum == fedMap.getMaxIndexInRange(1))
			return true;

		if(fedMap.getType() == FType.ROW) {
			return (rowNum == fedMap.getMaxIndexInRange(0) && (colNum == 1 || colNum == fedMap.getSize()))
				|| (colNum > 1 && rowNum == fedMap.getSize());
		}
		else if(fedMap.getType() == FType.COL) {
			return ((rowNum == 1 || rowNum == fedMap.getSize()) && colNum == fedMap.getMaxIndexInRange(1))
				|| (rowNum > 1 && colNum == fedMap.getSize());
		}
		throw new DMLRuntimeException("Only row partitioned or column partitioned federated input supported yet.");
	}

	private void setOutputCellwise(ExecutionContext ec, Future<FederatedResponse>[] response, FederationMap fedMap)
	{
		FType fedType = fedMap.getType();
		AggOp aggOp = ((SpoofCellwise)_op).getAggOp();
		CellType cellType = ((SpoofCellwise)_op).getCellType();
		if(cellType == CellType.FULL_AGG) { // full aggregation
			AggregateUnaryOperator aop = null;
			if(aggOp == AggOp.SUM || aggOp == AggOp.SUM_SQ) {
				// aggregate partial results from federated responses as sum
				aop = InstructionUtils.parseBasicAggregateUnaryOperator("uak+");
			}
			else if(aggOp == AggOp.MIN) {
				// aggregate partial results from federated responses as min
				aop = InstructionUtils.parseBasicAggregateUnaryOperator("uamin");
			}
			else if(aggOp == AggOp.MAX) {
				// aggregate partial results from federated responses as max
				aop = InstructionUtils.parseBasicAggregateUnaryOperator("uamax");
			}
			else {
				throw new DMLRuntimeException("Aggregation operation not supported yet.");
			}
			ec.setVariable(_output.getName(), FederationUtils.aggScalar(aop, response));
		}
		else if(cellType == CellType.ROW_AGG) { // row aggregation
			if(fedType == FType.ROW) {
				// bind partial results from federated responses
				ec.setMatrixOutput(_output.getName(), FederationUtils.bind(response, false));
			}
			else if(fedType == FType.COL) {
				AggregateUnaryOperator aop = null;
				if(aggOp == AggOp.SUM || aggOp == AggOp.SUM_SQ)
					aop = InstructionUtils.parseBasicAggregateUnaryOperator("uark+");
				else if(aggOp == AggOp.MIN)
					aop = InstructionUtils.parseBasicAggregateUnaryOperator("uarmin");
				else if(aggOp == AggOp.MAX)
					aop = InstructionUtils.parseBasicAggregateUnaryOperator("uarmax");
				else
					throw new DMLRuntimeException("Aggregation operation not supported yet.");
				ec.setMatrixOutput(_output.getName(), FederationUtils.aggMatrix(aop, response, fedMap));
			}
			else {
				throw new DMLRuntimeException("Aggregation type for federated spoof instructions not supported yet.");
			}
		}
		else if(cellType == CellType.COL_AGG) { // col aggregation
			if(fedType == FType.ROW) {
				AggregateUnaryOperator aop = null;
				if(aggOp == AggOp.SUM || aggOp == AggOp.SUM_SQ)
					aop = InstructionUtils.parseBasicAggregateUnaryOperator("uack+");
				else if(aggOp == AggOp.MIN)
					aop = InstructionUtils.parseBasicAggregateUnaryOperator("uacmin");
				else if(aggOp == AggOp.MAX)
					aop = InstructionUtils.parseBasicAggregateUnaryOperator("uacmax");
				else
					throw new DMLRuntimeException("Aggregation operation not supported yet.");
				ec.setMatrixOutput(_output.getName(), FederationUtils.aggMatrix(aop, response, fedMap));
			}
			else if(fedType == FType.COL) {
				// bind partial results from federated responses
				ec.setMatrixOutput(_output.getName(), FederationUtils.bind(response, true));
			}
			else {
				throw new DMLRuntimeException("Aggregation type for federated spoof instructions not supported yet.");
			}
		}
		else if(cellType == CellType.NO_AGG) { // no aggregation
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
		RowType rowType = ((SpoofRowwise)_op).getRowType();
		if(rowType == RowType.FULL_AGG) { // full aggregation
			// aggregate partial results from federated responses as sum
			AggregateUnaryOperator aop = InstructionUtils.parseBasicAggregateUnaryOperator("uak+");
			ec.setVariable(_output.getName(), FederationUtils.aggScalar(aop, response));
		}
		else if(rowType == RowType.ROW_AGG) { // row aggregation
			// aggregate partial results from federated responses as rowSum
			AggregateUnaryOperator aop = InstructionUtils.parseBasicAggregateUnaryOperator("uark+");
			ec.setMatrixOutput(_output.getName(), FederationUtils.aggMatrix(aop, response, fedMap));
		}
		else if(rowType == RowType.COL_AGG
			|| rowType == RowType.COL_AGG_T
			|| rowType == RowType.COL_AGG_B1
			|| rowType == RowType.COL_AGG_B1_T
			|| rowType == RowType.COL_AGG_B1R
			|| rowType == RowType.COL_AGG_CONST) { // col aggregation
			// aggregate partial results from federated responses as colSum
			AggregateUnaryOperator aop = InstructionUtils.parseBasicAggregateUnaryOperator("uack+");
			ec.setMatrixOutput(_output.getName(), FederationUtils.aggMatrix(aop, response, fedMap));
		}
		else if(rowType == RowType.NO_AGG
			|| rowType == RowType.NO_AGG_B1
			|| rowType == RowType.NO_AGG_CONST) { // no aggregation
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
	
	private void setOutputMultiAgg(ExecutionContext ec, Future<FederatedResponse>[] response, FederationMap fedMap)
	{
		MatrixBlock[] partRes = FederationUtils.getResults(response);
		SpoofCellwise.AggOp[] aggOps = ((SpoofMultiAggregate)_op).getAggOps();
		for(int counter = 1; counter < partRes.length; counter++) {
			SpoofMultiAggregate.aggregatePartialResults(aggOps, partRes[0], partRes[counter]);
		}
		ec.setMatrixOutput(_output.getName(), partRes[0]);
	}

	private void setOutputOuterProduct(ExecutionContext ec, Future<FederatedResponse>[] response, FederationMap fedMap)
	{
		FType fedType = fedMap.getType();
		OutProdType outProdType = ((SpoofOuterProduct)_op).getOuterProdType();
		if(outProdType == OutProdType.LEFT_OUTER_PRODUCT) {
			if(fedType == FType.ROW) {
				// aggregate partial results from federated responses as elementwise sum
				AggregateUnaryOperator aop = InstructionUtils.parseBasicAggregateUnaryOperator("uak+");
				ec.setMatrixOutput(_output.getName(), FederationUtils.aggMatrix(aop, response, fedMap));
			}
			else if(fedType == FType.COL) {
				// bind partial results from federated responses
				ec.setMatrixOutput(_output.getName(), FederationUtils.bind(response, false));
			}
			else {
				throw new DMLRuntimeException("Only row partitioned or column partitioned federated matrices supported yet.");
			}
		}
		else if(outProdType == OutProdType.RIGHT_OUTER_PRODUCT) {
			if(fedType == FType.ROW) {
				// bind partial results from federated responses
				ec.setMatrixOutput(_output.getName(), FederationUtils.bind(response, false));
			}
			else if(fedType == FType.COL) {
				// aggregate partial results from federated responses as elementwise sum
				AggregateUnaryOperator aop = InstructionUtils.parseBasicAggregateUnaryOperator("uak+");
				ec.setMatrixOutput(_output.getName(), FederationUtils.aggMatrix(aop, response, fedMap));
			}
			else {
				throw new DMLRuntimeException("Only row partitioned or column partitioned federated matrices supported yet.");
			}
		}
		else if(outProdType == OutProdType.CELLWISE_OUTER_PRODUCT) {
			if(fedType == FType.ROW) {
				// rbind partial results from federated responses
				ec.setMatrixOutput(_output.getName(), FederationUtils.bind(response, false));
			}
			else if(fedType == FType.COL) {
				// cbind partial results from federated responses
				ec.setMatrixOutput(_output.getName(), FederationUtils.bind(response, true));
			}
			else {
				throw new DMLRuntimeException("Only row partitioned or column partitioned federated matrices supported yet.");
			}
		}
		else if(outProdType == OutProdType.AGG_OUTER_PRODUCT) {
			// aggregate partial results from federated responses as sum
			AggregateUnaryOperator aop = InstructionUtils.parseBasicAggregateUnaryOperator("uak+");
			ec.setVariable(_output.getName(), FederationUtils.aggScalar(aop, response));
		}
		else {
			throw new DMLRuntimeException("Outer Product Type " + outProdType + " not supported yet.");
		}
	}

}
