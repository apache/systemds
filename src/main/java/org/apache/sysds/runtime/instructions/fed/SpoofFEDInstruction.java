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

import java.util.ArrayList;
import java.util.Collections;
import java.util.concurrent.Future;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.hops.fedplanner.FTypes.AlignType;
import org.apache.sysds.hops.fedplanner.FTypes.FType;
import org.apache.sysds.runtime.DMLRuntimeException;
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
import org.apache.sysds.runtime.controlprogram.federated.FederationUtils;
import org.apache.sysds.runtime.controlprogram.federated.MatrixLineagePair;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.instructions.cp.SpoofCPInstruction;
import org.apache.sysds.runtime.instructions.spark.SpoofSPInstruction;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;

public class SpoofFEDInstruction extends FEDInstruction
{
	private final SpoofOperator _op;
	private final CPOperand[] _inputs;
	private final CPOperand _output;

	private SpoofFEDInstruction(SpoofOperator op, CPOperand[] in,
		CPOperand out, String opcode, String instStr) {
		super(FEDInstruction.FEDType.SpoofFused, opcode, instStr);
		_op = op;
		_inputs = in;
		_output = out;
	}

	public static SpoofFEDInstruction parseInstruction(SpoofCPInstruction inst, ExecutionContext ec){
		Class<?> scla = inst.getOperatorClass().getSuperclass();
		if(((scla == SpoofCellwise.class || scla == SpoofMultiAggregate.class || scla == SpoofOuterProduct.class)
			&& SpoofFEDInstruction.isFederated(ec, inst.getInputs(), scla))
			|| (scla == SpoofRowwise.class && SpoofFEDInstruction.isFederated(ec, FType.ROW, inst.getInputs(), scla))) {
			return SpoofFEDInstruction.parseInstruction(inst);
		}
		return null;
	}

	public static SpoofFEDInstruction parseInstruction(SpoofSPInstruction inst, ExecutionContext ec){
		Class<?> scla = inst.getOperatorClass().getSuperclass();
		if(((scla == SpoofCellwise.class || scla == SpoofMultiAggregate.class || scla == SpoofOuterProduct.class)
			&& SpoofFEDInstruction.isFederated(ec, inst.getInputs(), scla))
			|| (scla == SpoofRowwise.class && SpoofFEDInstruction.isFederated(ec, FType.ROW, inst.getInputs(), scla))) {
			return SpoofFEDInstruction.parseInstruction(inst);
		}
		return null;
	}

	private static SpoofFEDInstruction parseInstruction(SpoofCPInstruction instr) {
		return new SpoofFEDInstruction(instr.getSpoofOperator(), instr.getInputs(), instr.getOutput(),
			instr.getOpcode(), instr.getInstructionString());
	}

	private static SpoofFEDInstruction parseInstruction(SpoofSPInstruction instr) {
		SpoofOperator op = CodegenUtils.createInstance(instr.getOperatorClass());
		return new SpoofFEDInstruction(op, instr.getInputs(), instr.getOutput(), instr.getOpcode(),
			instr.getInstructionString());
	}

	public static SpoofFEDInstruction parseInstruction(String str) {
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
	public void processInstruction(ExecutionContext ec) {
		FederationMap fedMap = null;
		for(CPOperand cpo : _inputs) { // searching for the first federated matrix to obtain the federation map
			Data tmpData = ec.getVariable(cpo);
			if(tmpData instanceof MatrixObject && ((MatrixObject)tmpData).isFederatedExcept(FType.BROADCAST)) {
				fedMap = ((MatrixObject)tmpData).getFedMapping();
				break;
			}
		}

		Class<?> scla = _op.getClass().getSuperclass();
		SpoofFEDType spoofType = null;
		if(scla == SpoofCellwise.class)
			spoofType = new SpoofFEDCellwise(_op, _output, fedMap.getType());
		else if(scla == SpoofRowwise.class)
			spoofType = new SpoofFEDRowwise(_op, _output, fedMap.getType());
		else if(scla == SpoofMultiAggregate.class)
			spoofType = new SpoofFEDMultiAgg(_op, _output, fedMap.getType());
		else if(scla == SpoofOuterProduct.class)
			spoofType = new SpoofFEDOuterProduct(_op, _output, fedMap.getType(), _inputs);
		else
			throw new DMLRuntimeException("Federated code generation only supported" +
				" for cellwise, rowwise, multiaggregate, and outerproduct templates.");

		processRequest(ec, fedMap, spoofType);
	}

	private void processRequest(ExecutionContext ec, FederationMap fedMap, SpoofFEDType spoofType) {
		ArrayList<FederatedRequest> frBroadcast = new ArrayList<>();
		ArrayList<FederatedRequest[]> frBroadcastSliced = new ArrayList<>();
		long[] frIds = new long[_inputs.length];
		int index = 0;

		for(CPOperand cpo : _inputs) {
			Data tmpData = ec.getVariable(cpo);
			if(tmpData instanceof MatrixObject) {
				MatrixLineagePair mo = MatrixLineagePair.of((MatrixObject) tmpData,
					DMLScript.LINEAGE ? ec.getLineageItem(cpo) : null);
				if(mo.isFederatedExcept(FType.BROADCAST)) {
					frIds[index++] = mo.getFedMapping().getID();
				}
				else if(spoofType.needsBroadcastSliced(fedMap, mo.getNumRows(), mo.getNumColumns(), index)) {
					FederatedRequest[] tmpFr = spoofType.broadcastSliced(mo, fedMap);
					frIds[index++] = tmpFr[0].getID();
					frBroadcastSliced.add(tmpFr);
				}
				else {
					FederatedRequest tmpFr = fedMap.broadcast(mo);
					frIds[index++] = tmpFr.getID();
					frBroadcast.add(tmpFr);
				}
			}
			else if(tmpData instanceof ScalarObject) {
				ScalarObject so = (ScalarObject) tmpData;
				FederatedRequest tmpFr = fedMap.broadcast(so);
				frIds[index++] = tmpFr.getID();
				frBroadcast.add(tmpFr);
			}
		}

		// change the is_literal flag from true to false because when broadcasted it is not a literal anymore
		instString = instString.replace("true", "false");

		FederatedRequest frCompute = FederationUtils.callInstruction(instString, _output, _inputs, frIds);

		FederatedRequest frGet = null;
		FederatedRequest frCleanup = null;
		if(!spoofType.isFedOutput()) {
			// get partial results from federated workers
			frGet = new FederatedRequest(RequestType.GET_VAR, frCompute.getID());
			// cleanup the federated request of callInstruction
			frCleanup = fedMap.cleanup(getTID(), frCompute.getID());
		}

		FederatedRequest[] frAll;
		if(frGet == null) // no get request if output is kept federated
			frAll = ArrayUtils.addAll(
				frBroadcast.toArray(new FederatedRequest[0]), frCompute);
		else
			frAll = ArrayUtils.addAll(
				frBroadcast.toArray(new FederatedRequest[0]), frCompute, frGet, frCleanup);

		Future<FederatedResponse>[] response = fedMap.executeMultipleSlices(
			getTID(), true, frBroadcastSliced.toArray(new FederatedRequest[0][]), frAll);

		// setting the output with respect to the different aggregation types
		// of the different spoof templates
		spoofType.setOutput(ec, response, fedMap, frCompute.getID());
	}


	// abstract class to differentiate between the different spoof templates
	private static abstract class SpoofFEDType {
		CPOperand _output;
		FType _fedType;

		protected SpoofFEDType(CPOperand out, FType fedType) {
			_output = out;
			_fedType = fedType;
		}

		/**
		 * performs the sliced broadcast of the given matrix object
		 *
		 * @param mo the matrix object to broadcast sliced with the respective lineage item
		 * @param fedMap the federated mapping
		 * @return FederatedRequest[] the resulting federated request array of the broadcast
		 */
		protected FederatedRequest[] broadcastSliced(MatrixLineagePair mo, FederationMap fedMap) {
			return fedMap.broadcastSliced(mo, false);
		}

		/**
		 * determine if a specific matrix object needs to be broadcast sliced
		 *
		 * @param fedMap the federated mapping
		 * @param rowNum the number of rows of the matrix object
		 * @param colNum the number of columns of the matrix object
		 * @param inputIndex the index of the matrix inside the instruction inputs
		 * @return boolean indicates if the matrix needs to be broadcast sliced
		 */
		protected boolean needsBroadcastSliced(FederationMap fedMap, long rowNum, long colNum, int inputIndex) {
			//TODO fix check by num rows/cols
			boolean retVal = (rowNum == fedMap.getMaxIndexInRange(0) && colNum == fedMap.getMaxIndexInRange(1));
			if(_fedType == FType.ROW)
				retVal |= (rowNum == fedMap.getMaxIndexInRange(0)
					&& (colNum == 1 || fedMap.getMaxIndexInRange(1) == 1));
			else if(_fedType == FType.COL)
				retVal |= (colNum == fedMap.getMaxIndexInRange(1)
					&& (rowNum == 1 || fedMap.getMaxIndexInRange(0) == 1));
			else {
				throw new DMLRuntimeException("Only row partitioned or column" +
					" partitioned federated input supported yet.");
			}
			return retVal;
		}

		/**
		 * set the output by either calling setFedOutput to keep the output federated
		 * or calling aggResult to aggregate the partial results locally
		 */
		protected void setOutput(ExecutionContext ec, Future<FederatedResponse>[] response,
			FederationMap fedMap, long frComputeID) {
			if(isFedOutput())
				setFedOutput(ec, fedMap, frComputeID);
			else
				aggResult(ec, response, fedMap);
		}

		// determine if the output can be kept on the federated sites
		protected abstract boolean isFedOutput();
		// set the output by deriving new a federated mapping
		protected abstract void setFedOutput(ExecutionContext ec, FederationMap fedMap, long frComputeID);
		// aggregate the partial results locally
		protected abstract void aggResult(ExecutionContext ec, Future<FederatedResponse>[] response,
			FederationMap fedMap);
	}

	// CELLWISE TEMPLATE
	private static class SpoofFEDCellwise extends SpoofFEDType {
		private final SpoofCellwise _op;
		private final CellType _cellType;

		SpoofFEDCellwise(SpoofOperator op, CPOperand out, FType fedType) {
			super(out, fedType);
			_op = (SpoofCellwise)op;
			_cellType = _op.getCellType();
		}

		protected boolean isFedOutput() {
			boolean retVal = false;
			retVal |= (_cellType == CellType.ROW_AGG && _fedType == FType.ROW);
			retVal |= (_cellType == CellType.COL_AGG && _fedType == FType.COL);
			retVal |= (_cellType == CellType.NO_AGG);
			return retVal;
		}

		protected void setFedOutput(ExecutionContext ec, FederationMap fedMap, long frComputeID) {
			// derive output federated mapping
			MatrixObject out = ec.getMatrixObject(_output);
			FederationMap newFedMap = modifyFedRanges(fedMap.copyWithNewID(frComputeID));
			out.setFedMapping(newFedMap);
		}

		private FederationMap modifyFedRanges(FederationMap fedMap) {
			if(_cellType == CellType.ROW_AGG || _cellType == CellType.COL_AGG) {
				int dim = (_cellType == CellType.COL_AGG ? 0 : 1);
				// crop federation map to a vector
				fedMap.modifyFedRanges(1, dim);
			}
			return fedMap;
		}

		protected void aggResult(ExecutionContext ec, Future<FederatedResponse>[] response,
			FederationMap fedMap) {
			AggOp aggOp = _op.getAggOp();

			// build up the instruction for aggregation
			// (uak+/uamin/uamax/uark+/uarmin/uarmax/uack+/uacmin/uacmax)
			String aggInst = "ua";
			switch(_cellType) {
				case FULL_AGG: break;
				case ROW_AGG: aggInst += "r"; break;
				case COL_AGG: aggInst += "c"; break;
				case NO_AGG:
				default:
					throw new DMLRuntimeException("Aggregation type not supported yet.");
			}

			switch(aggOp) {
				case SUM:
				case SUM_SQ: aggInst += "k+"; break;
				case MIN:    aggInst += "min"; break;
				case MAX:    aggInst += "max"; break;
				default:
					throw new DMLRuntimeException("Aggregation operation not supported yet.");
			}

			AggregateUnaryOperator aop = InstructionUtils.parseBasicAggregateUnaryOperator(aggInst);
			if(_cellType == CellType.FULL_AGG)
				ec.setVariable(_output.getName(), FederationUtils.aggScalar(aop, response));
			else
				ec.setMatrixOutput(_output.getName(), FederationUtils.aggMatrix(aop, response, fedMap));
		}
	}

	// ROWWISE TEMPLATE
	private static class SpoofFEDRowwise extends SpoofFEDType {
		private final SpoofRowwise _op;
		private final RowType _rowType;

		SpoofFEDRowwise(SpoofOperator op, CPOperand out, FType fedType) {
			super(out, fedType);
			_op = (SpoofRowwise)op;
			_rowType = _op.getRowType();
		}

		protected boolean isFedOutput() {
			boolean retVal = false;
			retVal |= (_rowType == RowType.NO_AGG);
			retVal |= (_rowType == RowType.NO_AGG_B1);
			retVal |= (_rowType == RowType.NO_AGG_CONST);
			retVal &= (_fedType == FType.ROW);
			return retVal;
		}

		protected void setFedOutput(ExecutionContext ec, FederationMap fedMap, long frComputeID) {
			// derive output federated mapping
			MatrixObject out = ec.getMatrixObject(_output);
			FederationMap newFedMap = fedMap.copyWithNewID(frComputeID).modifyFedRanges(out.getNumColumns(), 1);
			out.setFedMapping(newFedMap);
		}

		protected void aggResult(ExecutionContext ec, Future<FederatedResponse>[] response,
			FederationMap fedMap) {
			if(_fedType != FType.ROW)
				throw new DMLRuntimeException("Only row partitioned federated matrices supported yet.");

			// build up the instruction for aggregation (uak+/uark+/uack+)
			String aggInst = "ua";
			if(_rowType == RowType.FULL_AGG) // full aggregation
				aggInst += "k+";
			else if(_rowType == RowType.ROW_AGG) // row aggregation
				aggInst += "rk+";
			else if(_rowType.isColumnAgg()) // col aggregation
				aggInst += "ck+";
			else
				throw new DMLRuntimeException("AggregationType not supported yet.");

			// aggregate partial results from federated responses as sum/rowSum/colSum
			AggregateUnaryOperator aop = InstructionUtils.parseBasicAggregateUnaryOperator(aggInst);
			if(_rowType == RowType.FULL_AGG)
				ec.setVariable(_output.getName(), FederationUtils.aggScalar(aop, response));
			else
				ec.setMatrixOutput(_output.getName(), FederationUtils.aggMatrix(aop, response, fedMap));
		}
	}

	// MULTIAGGREGATE TEMPLATE
	private static class SpoofFEDMultiAgg extends SpoofFEDType {
		private final SpoofMultiAggregate _op;

		SpoofFEDMultiAgg(SpoofOperator op, CPOperand out, FType fedType) {
			super(out, fedType);
			_op = (SpoofMultiAggregate)op;
		}

		protected boolean isFedOutput() {
			return false;
		}

		protected void setFedOutput(ExecutionContext ec, FederationMap fedMap, long frComputeID) {
			throw new DMLRuntimeException("SpoofFEDMultiAgg cannot create a federated output.");
		}

		protected void aggResult(ExecutionContext ec, Future<FederatedResponse>[] response,
			FederationMap fedMap) {
				MatrixBlock[] partRes = FederationUtils.getResults(response);
				SpoofCellwise.AggOp[] aggOps = _op.getAggOps();
				for(int counter = 1; counter < partRes.length; counter++) {
					SpoofMultiAggregate.aggregatePartialResults(aggOps, partRes[0], partRes[counter]);
				}
				ec.setMatrixOutput(_output.getName(), partRes[0]);
			}
	}

	// OUTER PRODUCT TEMPLATE
	private static class SpoofFEDOuterProduct extends SpoofFEDType {
		private final SpoofOuterProduct _op;
		private final OutProdType _outProdType;
		private CPOperand[] _inputs;

		SpoofFEDOuterProduct(SpoofOperator op, CPOperand out, FType fedType, CPOperand[] inputs) {
			super(out, fedType);
			_op = (SpoofOuterProduct)op;
			_outProdType = _op.getOuterProdType();
			_inputs = inputs;
		}

		@Override
		protected FederatedRequest[] broadcastSliced(MatrixLineagePair mo, FederationMap fedMap) {
			return fedMap.broadcastSliced(mo, (_fedType == FType.COL));
		}

		protected boolean needsBroadcastSliced(FederationMap fedMap, long rowNum, long colNum, int inputIndex) {
			boolean retVal = false;
			
			retVal |= (rowNum == fedMap.getMaxIndexInRange(0) && colNum == fedMap.getMaxIndexInRange(1));
			
			if(_fedType == FType.ROW)
				retVal |= (rowNum == fedMap.getMaxIndexInRange(0)) && (inputIndex != 2); // input at index 2 is V
			else if(_fedType == FType.COL)
				retVal |= (rowNum == fedMap.getMaxIndexInRange(1)) && (inputIndex != 1); // input at index 1 is U
			else
				throw new DMLRuntimeException("Only row partitioned or column" +
					" partitioned federated input supported yet.");

			return retVal;
		}

		protected boolean isFedOutput() {
			boolean retVal = false;
			retVal |= (_outProdType == OutProdType.LEFT_OUTER_PRODUCT && _fedType == FType.COL);
			retVal |= (_outProdType == OutProdType.RIGHT_OUTER_PRODUCT && _fedType == FType.ROW);
			retVal |= (_outProdType == OutProdType.CELLWISE_OUTER_PRODUCT);
			return retVal;
		}

		protected void setFedOutput(ExecutionContext ec, FederationMap fedMap, long frComputeID) {
			FederationMap newFedMap = fedMap.copyWithNewID(frComputeID);
			long[] outDims = new long[2];

			// find the resulting output dimensions
			MatrixObject X = ec.getMatrixObject(_inputs[0]);
			switch(_outProdType) {
				case LEFT_OUTER_PRODUCT: // LEFT: nrows of transposed X, ncols of U
					newFedMap = newFedMap.transpose();
					outDims[0] = X.getNumColumns();
					outDims[1] = ec.getMatrixObject(_inputs[1]).getNumColumns();
					break;
				case RIGHT_OUTER_PRODUCT: // RIGHT: nrows of X, ncols of V
					outDims[0] = X.getNumRows();
					outDims[1] = ec.getMatrixObject(_inputs[2]).getNumColumns();
					break;
				case CELLWISE_OUTER_PRODUCT: // BASIC: preserve dimensions of X
					outDims[0] = X.getNumRows();
					outDims[1] = X.getNumColumns();
					break;
				default:
					throw new DMLRuntimeException("Outer Product Type " + _outProdType + " not supported yet.");
			}

			// derive output federated mapping
			MatrixObject out = ec.getMatrixObject(_output);
			int dim = (newFedMap.getType() == FType.ROW ? 1 : 0);
			newFedMap.modifyFedRanges(outDims[dim], dim);
			out.setFedMapping(newFedMap);
		}

		protected void aggResult(ExecutionContext ec, Future<FederatedResponse>[] response,
			FederationMap fedMap) {
			AggregateUnaryOperator aop = InstructionUtils.parseBasicAggregateUnaryOperator("uak+");
			switch(_outProdType) {
				case LEFT_OUTER_PRODUCT:
				case RIGHT_OUTER_PRODUCT:
					// aggregate partial results from federated responses as elementwise sum
					ec.setMatrixOutput(_output.getName(), FederationUtils.aggMatrix(aop, response, fedMap));
					break;
				case AGG_OUTER_PRODUCT:
					// aggregate partial results from federated responses as sum
					ec.setVariable(_output.getName(), FederationUtils.aggScalar(aop, response));
					break;
				default:
					throw new DMLRuntimeException("Outer Product Type " + _outProdType + " not supported yet.");
			}
		}
	}

	public static boolean isFederated(ExecutionContext ec, CPOperand[] inputs, Class<?> scla) {
		// TODO: use the `SpoofOperator` instead of the `Class<?>`, as it is more robust to inheritance hierarchy change
		return isFederated(ec, null, inputs, scla);
	}

	public static boolean isFederated(ExecutionContext ec, FType type, CPOperand[] inputs, Class<?> scla) {
		FederationMap fedMap = null;
		boolean retVal = false;

		ArrayList<AlignType> alignmentTypes = new ArrayList<>();

		for(CPOperand input : inputs) {
			Data data = ec.getVariable(input);
			if(data instanceof MatrixObject && ((MatrixObject) data).isFederated(type)
				&& !((MatrixObject) data).isFederated(FType.BROADCAST)) {
				MatrixObject mo = ((MatrixObject) data);
				if(fedMap == null) { // first federated matrix
					fedMap = mo.getFedMapping();
					retVal = true;

					// setting the alignment types for alignment check on further federated matrices
					alignmentTypes.add(mo.isFederated(FType.ROW) ? AlignType.ROW : AlignType.COL);
					if(scla == SpoofOuterProduct.class)
						Collections.addAll(alignmentTypes, AlignType.ROW_T, AlignType.COL_T);
				}
				else if(!fedMap.isAligned(mo.getFedMapping(), alignmentTypes.toArray(new AlignType[0]))) {
					retVal = false; // multiple federated matrices must be aligned
				}
			}
		}
		return retVal;
	}
}
