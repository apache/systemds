/*
 * Copyright 2019 Graz University of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tugraz.sysds.runtime.instructions.fed;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Future;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.tugraz.sysds.common.Types.DataType;
import org.tugraz.sysds.common.Types.ValueType;
import org.tugraz.sysds.runtime.DMLRuntimeException;
import org.tugraz.sysds.runtime.controlprogram.caching.MatrixObject;
import org.tugraz.sysds.runtime.controlprogram.context.ExecutionContext;
import org.tugraz.sysds.runtime.controlprogram.federated.FederatedData;
import org.tugraz.sysds.runtime.controlprogram.federated.FederatedRange;
import org.tugraz.sysds.runtime.controlprogram.federated.FederatedRequest;
import org.tugraz.sysds.runtime.controlprogram.federated.FederatedResponse;
import org.tugraz.sysds.runtime.functionobjects.Multiply;
import org.tugraz.sysds.runtime.functionobjects.Plus;
import org.tugraz.sysds.runtime.instructions.InstructionUtils;
import org.tugraz.sysds.runtime.instructions.cp.CPOperand;
import org.tugraz.sysds.runtime.matrix.data.MatrixBlock;
import org.tugraz.sysds.runtime.matrix.operators.AggregateBinaryOperator;
import org.tugraz.sysds.runtime.matrix.operators.AggregateOperator;
import org.tugraz.sysds.runtime.matrix.operators.Operator;

public class AggregateBinaryFEDInstruction extends BinaryFEDInstruction {
	
	public AggregateBinaryFEDInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out, String opcode,
			String istr) {
		super(FEDType.AggregateBinary, op, in1, in2, out, opcode, istr);
	}
	
	public static AggregateBinaryFEDInstruction parseInstruction(String str) {
		CPOperand in1 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand in2 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		
		if( !opcode.equalsIgnoreCase("ba+*") ) {
			throw new DMLRuntimeException("AggregateBinaryInstruction.parseInstruction():: Unknown opcode " + opcode);
		}
		
		InstructionUtils.checkNumFields(parts, 4);
		in1.split(parts[1]);
		in2.split(parts[2]);
		out.split(parts[3]);
		int k = Integer.parseInt(parts[4]);
		
		AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
		AggregateBinaryOperator aggbin = new AggregateBinaryOperator(Multiply.getMultiplyFnObject(), agg, k);
		return new AggregateBinaryFEDInstruction(aggbin, in1, in2, out, opcode, str);
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) {
		//get inputs
		MatrixObject mo1 = ec.getMatrixObject(input1.getName());
		MatrixObject mo2 = ec.getMatrixObject(input2.getName());
		MatrixObject out = ec.getMatrixObject(output.getName());
		
		// TODO compute matrix multiplication
		// compute matrix-vector multiplication
		AggregateBinaryOperator ab_op = (AggregateBinaryOperator) _optr;
		if (mo1.isFederated())
			federatedAggregateBinary(mo1, mo2, out, ab_op, false);
		else if (mo2.isFederated())
			federatedAggregateBinary(mo2, mo1, out, ab_op, true);
		else
			throw new DMLRuntimeException("Federated instruction on non federated data");
	}
	
	/**
	 * Performs a federated binary aggregation (currently only MV and VM is supported).
	 * 
	 * @param mo1 the federated matrix object
	 * @param mo2 the other matrix object (currently has to be a vector)
	 * @param op the operation
	 * @param swapParams if <code>this</code> and <code>other</code> should be swapped (other op this, instead of
	 *                   this op other)
	 */
	public static void federatedAggregateBinary(MatrixObject mo1, MatrixObject mo2, MatrixObject output,
		AggregateBinaryOperator op, boolean swapParams)
	{
		if( !(op.binaryFn instanceof Multiply && op.aggOp.increOp.fn instanceof Plus) )
			throw new DMLRuntimeException("Only matrix-vector is supported for federated binary aggregation");
		// create output matrix
		MatrixBlock resultBlock;
		// if we chang the order of parameters, so VM instead of MV, the output has different dimensions
		if( swapParams ) {
			output.getDataCharacteristics().setRows(1).setCols(mo1.getNumColumns());
			resultBlock = new MatrixBlock(1, (int) mo1.getNumColumns(), false);
		}
		else {
			output.getDataCharacteristics().setRows(mo1.getNumRows()).setCols(1);
			resultBlock = new MatrixBlock((int) mo1.getNumRows(), 1, false);
		}
		List<Pair<FederatedRange, Future<FederatedResponse>>> idResponsePairs = new ArrayList<>();
		MatrixBlock vector = mo2.acquireRead();
		for (Map.Entry<FederatedRange, FederatedData> entry : mo1.getFedMapping().entrySet()) {
			FederatedRange range = entry.getKey();
			FederatedData fedData = entry.getValue();
			if( !fedData.isInitialized() ) {
				throw new DMLRuntimeException("Not all FederatedData was initialized for federated matrix");
			}
			int[] beginDimsInt = range.getBeginDimsInt();
			int[] endDimsInt = range.getEndDimsInt();
			// params for federated request
			List<Object> params = new ArrayList<>();
			// we broadcast the needed part of the small vector
			MatrixBlock vectorSlice;
			if( swapParams ) {
				vectorSlice = new MatrixBlock(1, endDimsInt[0] - beginDimsInt[0], false);
				vector.slice(0, 0, beginDimsInt[0], endDimsInt[0] - 1, vectorSlice);
			}
			else {
				vectorSlice = new MatrixBlock(endDimsInt[1] - beginDimsInt[1], 1, false);
				vector.slice(beginDimsInt[1], endDimsInt[1] - 1, 0, 0, vectorSlice);
			}
			params.add(vectorSlice);
			params.add(!swapParams); // if is matrix vector multiplication true, otherwise false
			Future<FederatedResponse> future = fedData.executeFederatedOperation(
					new FederatedRequest(FederatedRequest.FedMethod.MATVECMULT, params), true);
			idResponsePairs.add(new ImmutablePair<>(range, future));
		}
		mo2.release();
		try {
			for (Pair<FederatedRange, Future<FederatedResponse>> idResponsePair : idResponsePairs) {
				FederatedRange range = idResponsePair.getLeft();
				FederatedResponse federatedResponse = idResponsePair.getRight().get();
				int[] beginDims = range.getBeginDimsInt();
				MatrixBlock mb = (MatrixBlock) federatedResponse.getData();
				// TODO performance optimizations
				// TODO Improve Vector Matrix multiplication accuracy
				// Add worker response to resultBlock
				for (int r = 0; r < mb.getNumRows(); r++)
					for (int c = 0; c < mb.getNumColumns(); c++) {
						int resultRow = r + (swapParams ? 0 : beginDims[0]);
						int resultColumn = c + (swapParams ? beginDims[1] : 0);
						resultBlock.quickSetValue(resultRow, resultColumn,
								resultBlock.quickGetValue(resultRow, resultColumn) + mb.quickGetValue(r, c));
					}
			}
		}
		catch (Exception e) {
			throw new DMLRuntimeException("Federated binary aggregation failed", e);
		}
		// wait for all data to be written to resultBlock
		long nnz = resultBlock.recomputeNonZeros();
		output.acquireModify(resultBlock);
		output.getDataCharacteristics().setNonZeros(nnz);
		output.release();
	}
}
