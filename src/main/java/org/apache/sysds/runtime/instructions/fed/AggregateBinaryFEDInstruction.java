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

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.MutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.FederatedData;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRange;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.AggregateBinaryOperator;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.util.CommonThreadPool;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

public class AggregateBinaryFEDInstruction extends BinaryFEDInstruction {
	
	public AggregateBinaryFEDInstruction(Operator op, CPOperand in1,
		CPOperand in2, CPOperand out, String opcode, String istr) {
		super(FEDType.AggregateBinary, op, in1, in2, out, opcode, istr);
	}
	
	public static AggregateBinaryFEDInstruction parseInstruction(String str) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		
		if(!opcode.equalsIgnoreCase("ba+*")) {
			throw new DMLRuntimeException("AggregateBinaryInstruction.parseInstruction():: Unknown opcode " + opcode);
		}
		
		InstructionUtils.checkNumFields(parts, 4);
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand in2 = new CPOperand(parts[2]);
		CPOperand out = new CPOperand(parts[3]);
		int k = Integer.parseInt(parts[4]);
		
		return new AggregateBinaryFEDInstruction(
			InstructionUtils.getMatMultOperator(k), in1, in2, out, opcode, str);
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) {
		//get inputs
		MatrixObject mo1 = ec.getMatrixObject(input1.getName());
		MatrixObject mo2 = ec.getMatrixObject(input2.getName());
		MatrixObject out = ec.getMatrixObject(output.getName());
		
		// compute matrix-vector multiplication
		AggregateBinaryOperator ab_op = (AggregateBinaryOperator) _optr;
		if (mo1.isFederated() && mo2.getNumColumns() == 1) {// MV
			MatrixBlock vector = mo2.acquireRead();
			federatedAggregateBinaryMV(mo1, vector, out, ab_op, true);
			mo2.release();
		}
		else if (mo2.isFederated() && mo1.getNumRows() == 1) {// VM
			MatrixBlock vector = mo1.acquireRead();
			federatedAggregateBinaryMV(mo2, vector, out, ab_op, false);
			mo1.release();
		}
		else // MM
			federatedAggregateBinary(mo1, mo2, out);
	}
	
	/**
	 * Performs a federated binary aggregation (currently only MV and VM is supported).
	 *
	 * @param mo1 the first matrix object
	 * @param mo2 the other matrix object
	 * @param out output matrix object
	 */
	private static void federatedAggregateBinary(MatrixObject mo1, MatrixObject mo2, MatrixObject out) {
		boolean distributeCols = false;
		// if distributeCols = true we distribute cols of mo2 and do a MV multiplications, otherwise we
		// distribute rows of mo1 and do VM multiplications
		if (mo1.isFederated() && mo2.isFederated()) {
			// both are federated -> distribute smaller matrix
			// TODO do more in depth checks like: how many federated workers, how big is the actual data we send and so on
			// maybe once we track number of non zeros we could use that to get a better estimation of how much data
			// will be requested?
			distributeCols = mo2.getNumColumns() * mo2.getNumRows() < mo1.getNumColumns() * mo1.getNumRows();
		}
		else if (mo2.isFederated() && !mo1.isFederated()) {
			// Distribute mo1 which is not federated
			distributeCols = true;
		}
		// TODO performance if both matrices are federated
		Map<FederatedRange, FederatedData> mapping = distributeCols ? mo1.getFedMapping() : mo2.getFedMapping();
		MatrixBlock matrixBlock = distributeCols ? mo2.acquireRead() : mo1.acquireRead();
		ExecutorService pool = CommonThreadPool.get(mapping.size());
		ArrayList<Pair<FederatedRange, MatrixBlock>> results = new ArrayList<>();
		ArrayList<FederatedMMTask> tasks = new ArrayList<>();
		for (Map.Entry<FederatedRange, FederatedData> fedMap : mapping.entrySet()) {
			// this resultPair will contain both position of partial result and the partial result itself of the operations
			MutablePair<FederatedRange, MatrixBlock> resultPair = new MutablePair<>();
			// they all get references to the real block, the task slices out the needed part and does the
			// multiplication, therefore they can share the object since we use it immutably
			tasks.add(new FederatedMMTask(fedMap.getKey(), fedMap.getValue(), resultPair, matrixBlock, distributeCols));
			results.add(resultPair);
		}
		CommonThreadPool.invokeAndShutdown(pool, tasks);
		(distributeCols?mo2:mo1).release();
		
		// combine results
		if (mo1.getNumRows() > Integer.MAX_VALUE || mo2.getNumColumns() > Integer.MAX_VALUE) {
			throw new DMLRuntimeException("Federated matrix is too large for federated distribution");
		}
		out.acquireModify(combinePartialMMResults(results, (int) mo1.getNumRows(), (int) mo2.getNumColumns()));
		out.release();
	}
	
	private static MatrixBlock combinePartialMMResults(ArrayList<Pair<FederatedRange, MatrixBlock>> results, 
		int rows, int cols) {
		// TODO support large blocks with > int size
		MatrixBlock resultBlock = new MatrixBlock(rows, cols, false);
		for (Pair<FederatedRange, MatrixBlock> partialResult : results) {
			FederatedRange range = partialResult.getLeft();
			MatrixBlock partialBlock = partialResult.getRight();
			int[] dimsLower = range.getBeginDimsInt();
			int[] dimsUpper = range.getEndDimsInt();
			resultBlock.copy(dimsLower[0], dimsUpper[0] - 1, dimsLower[1], dimsUpper[1] - 1, partialBlock, false);
		}
		resultBlock.recomputeNonZeros();
		return resultBlock;
	}
	
	/**
	 * Performs a federated binary aggregation on a Matrix and Vector.
	 *
	 * @param fedMo          the federated matrix object
	 * @param vector         the vector
	 * @param output         the output matrix object
	 * @param op             the operation
	 * @param matrixVectorOp true if matrix vector operation, false if vector matrix op
	 */
	public static void federatedAggregateBinaryMV(MatrixObject fedMo, MatrixBlock vector, MatrixObject output,
			AggregateBinaryOperator op, boolean matrixVectorOp) {
		if (!(op.binaryFn instanceof Multiply && op.aggOp.increOp.fn instanceof Plus))
			throw new DMLRuntimeException("Only matrix-vector is supported for federated binary aggregation");
		// fixed implementation only for mv, vm multiply and plus
		// TODO move this to a Lib class?
		
		// create output matrix
		MatrixBlock resultBlock;
		// if we change the order of parameters, so VM instead of MV, the output has different dimensions
		if (!matrixVectorOp) {
			output.getDataCharacteristics().setRows(1).setCols(fedMo.getNumColumns());
			resultBlock = new MatrixBlock(1, (int) fedMo.getNumColumns(), false);
		}
		else {
			output.getDataCharacteristics().setRows(fedMo.getNumRows()).setCols(1);
			resultBlock = new MatrixBlock((int) fedMo.getNumRows(), 1, false);
		}
		List<Pair<FederatedRange, Future<FederatedResponse>>> idResponsePairs = new ArrayList<>();
		// TODO parallel for loop (like on lines 125-136)
		for (Map.Entry<FederatedRange, FederatedData> entry : fedMo.getFedMapping().entrySet()) {
			FederatedRange range = entry.getKey();
			FederatedData fedData = entry.getValue();
			Future<FederatedResponse> future = executeMVMultiply(range, fedData, vector, matrixVectorOp);
			idResponsePairs.add(new ImmutablePair<>(range, future));
		}
		try {
			for (Pair<FederatedRange, Future<FederatedResponse>> idResponsePair : idResponsePairs) {
				FederatedRange range = idResponsePair.getLeft();
				FederatedResponse federatedResponse = idResponsePair.getRight().get();
				combinePartialMVResults(range, federatedResponse, resultBlock, matrixVectorOp);
			}
		}
		catch (Exception e) {
			throw new DMLRuntimeException("Federated binary aggregation failed", e);
		}
		long nnz = resultBlock.recomputeNonZeros();
		output.acquireModify(resultBlock);
		output.getDataCharacteristics().setNonZeros(nnz);
		output.release();
	}
	
	private static void combinePartialMVResults(FederatedRange range,
		FederatedResponse federatedResponse, MatrixBlock resultBlock, boolean matrixVectorOp)
	{
		try {
			int[] beginDims = range.getBeginDimsInt();
			MatrixBlock mb = (MatrixBlock) federatedResponse.getData()[0];
			// TODO performance optimizations
			// TODO Improve Vector Matrix multiplication accuracy: An idea would be to make use of kahan plus here,
			//  this should improve accuracy a bit, although we still lose out on the small error lost on the worker
			//  without having to return twice the amount of data (value + sum error)
			// Add worker response to resultBlock
			for (int r = 0; r < mb.getNumRows(); r++)
				for (int c = 0; c < mb.getNumColumns(); c++) {
					int resultRow = r + (!matrixVectorOp ? 0 : beginDims[0]);
					int resultColumn = c + (!matrixVectorOp ? beginDims[1] : 0);
					resultBlock.quickSetValue(resultRow, resultColumn,
						resultBlock.quickGetValue(resultRow, resultColumn) + mb.quickGetValue(r, c));
				}
		} catch (Exception e){
			throw new DMLRuntimeException("Combine partial results from federated matrix failed.", e);
		}
	}
	
	private static Future<FederatedResponse> executeMVMultiply(FederatedRange range,
		FederatedData fedData, MatrixBlock vector, boolean matrixVectorOp)
	{
		if (!fedData.isInitialized()) {
			throw new DMLRuntimeException("Not all FederatedData was initialized for federated matrix");
		}
		int[] beginDimsInt = range.getBeginDimsInt();
		int[] endDimsInt = range.getEndDimsInt();
		// params for federated request
		List<Object> params = new ArrayList<>();
		// we broadcast the needed part of the small vector
		MatrixBlock vectorSlice;
		if (!matrixVectorOp) {
			// if we the size already is ok, we do not have to copy a slice
			int length = endDimsInt[0] - beginDimsInt[0];
			if (vector.getNumColumns() == length) {
				vectorSlice = vector;
			}
			else {
				vectorSlice = new MatrixBlock(1, length, false);
				vector.slice(0, 0, beginDimsInt[0], endDimsInt[0] - 1, vectorSlice);
			}
		}
		else {
			int length = endDimsInt[1] - beginDimsInt[1];
			if (vector.getNumRows() == length) {
				vectorSlice = vector;
			}
			else {
				vectorSlice = new MatrixBlock(length, 1, false);
				vector.slice(beginDimsInt[1], endDimsInt[1] - 1, 0, 0, vectorSlice);
			}
		}
		params.add(vectorSlice);
		params.add(matrixVectorOp); // if is matrix vector multiplication true, otherwise false
		return fedData.executeFederatedOperation(
			new FederatedRequest(FederatedRequest.FedMethod.MATVECMULT, params), true);
	}
	
	private static class FederatedMMTask implements Callable<Void> {
		private FederatedRange _range;
		private FederatedData _data;
		private MutablePair<FederatedRange, MatrixBlock> _result;
		private MatrixBlock _otherMatrix;
		private boolean _distributeCols;
		
		public FederatedMMTask(FederatedRange range, FederatedData fedData,
			MutablePair<FederatedRange, MatrixBlock> result, MatrixBlock otherMatrix, boolean distributeCols)
		{
			_range = range;
			_data = fedData;
			_result = result;
			_otherMatrix = otherMatrix;
			_distributeCols = distributeCols;
		}
		
		@Override
		public Void call() throws Exception {
			if (_distributeCols)
				executeColWiseMVMultiplication();
			else
				executeRowWiseVMMultiplications();
			return null;
		}
		
		/**
		 * Distribute the non or smaller federated block as row vectors to the federated worker and do row number of
		 * times a vector-matrix multiplication. Non or smaller federated block is left operand.
		 *
		 * @throws InterruptedException if .get() on federated response future fails -> interrupted
		 * @throws ExecutionException   if .get() on federated response future fails -> execution failed
		 */
		private void executeRowWiseVMMultiplications() throws InterruptedException, ExecutionException {
			MatrixBlock result;
			// TODO support large matrices with long indexes
			int[] beginDims = _range.getBeginDimsInt();
			int[] endDims = _range.getEndDimsInt();
			// we take all rows but only the columns between the rows of the federated block of the other block (left
			// hand side of the calculation).
			int rowsBeginOtherBlock = 0;
			int colsBeginOtherBlock = beginDims[0];
			int rowsEndOtherBlock = _otherMatrix.getNumRows();
			int colsEndOtherBlock = endDims[0];
			// Size of partial result block for vm is rows of otherBlock * cols of federatedData
			result = new MatrixBlock(rowsEndOtherBlock - rowsBeginOtherBlock, endDims[1] - beginDims[1], false);
			// Set range of output in result block, rows are the number of rows of the other block, while columns
			// are the number of columns of our federated data
			_result.setLeft(new FederatedRange(new long[] {rowsBeginOtherBlock, beginDims[1]},
					new long[] {rowsEndOtherBlock, endDims[1]}));
			// vector which we will distribute otherBlock.rows number of times
			MatrixBlock vec = new MatrixBlock(1, colsEndOtherBlock - colsBeginOtherBlock, false);
			for (int r = rowsBeginOtherBlock; r < rowsEndOtherBlock; r++) {
				// slice row vector out of other matrix which we will send to federated worker
				_otherMatrix.slice(r, r, colsBeginOtherBlock, colsEndOtherBlock - 1, vec);
				// TODO experiment if sending multiple requests at the same time to the same worker increases
				//  performance (remove get and do multithreaded?)
				FederatedResponse response = executeMVMultiply(_range, _data, vec, _distributeCols).get();
				try{
					result.copy(r, r, 0, endDims[1] - beginDims[1] - 1, (MatrixBlock) response.getData()[0], true);
				} catch (Exception e) {
					throw new DMLRuntimeException(
						"Federated Matrix-Matrix Multiplication failed: ", e);
				}		
			}
			_result.setRight(result);
		}
		
		/**
		 * Distribute the non or smaller federated block as col vectors to the federated worker and do column number of
		 * times a matrix-vector multiplication. Non or smaller federated block is right operand.
		 *
		 * @throws InterruptedException if .get() on federated response future fails -> interrupted
		 * @throws ExecutionException   if .get() on federated response future fails -> execution failed
		 */
		private void executeColWiseMVMultiplication()
				throws InterruptedException, ExecutionException {
			MatrixBlock result;
			// TODO support large matrices with long indexes
			int[] beginDims = _range.getBeginDimsInt();
			int[] endDims = _range.getEndDimsInt();
			// we take all columns but only the rows between the columns of the federated block of the other block (right
			// hand side of the calculation).
			int rowsBeginOtherBlock = beginDims[1];
			int colsBeginOtherBlock = 0;
			int rowsEndOtherBlock = endDims[1];
			int colsEndOtherBlock = _otherMatrix.getNumColumns();
			// Size of partial result block for mv is rows of federated block * cols of other block
			result = new MatrixBlock(endDims[0] - beginDims[0], colsEndOtherBlock - colsBeginOtherBlock, false);
			// Set range of output in result block, rows are the number of rows of the federated data, while columns
			// are the number of columns of the other block
			_result.setLeft(new FederatedRange(new long[] {beginDims[0], colsBeginOtherBlock},
					new long[] {endDims[0], colsEndOtherBlock}));
			// vector which we will distribute otherBlock.cols number of times
			MatrixBlock vec = new MatrixBlock(rowsEndOtherBlock - rowsBeginOtherBlock, 1, false);
			for (int c = colsBeginOtherBlock; c < colsEndOtherBlock; c++) {
				// slice column vector out of other matrix which we will send to federated worker
				_otherMatrix.slice(rowsBeginOtherBlock, rowsEndOtherBlock - 1, c, c, vec);
				// TODO experiment if sending multiple requests at the same time to the same worker increases
				//  performance
				FederatedResponse response = executeMVMultiply(_range, _data, vec, _distributeCols).get();
				try {
					result.copy(0, endDims[0] - beginDims[0] - 1, c, c, (MatrixBlock) response.getData()[0], true);
				} catch (Exception e){
					throw new DMLRuntimeException(
						"Federated Matrix-Matrix Multiplication failed: ", e);
				}
				
			}
			_result.setRight(result);
		}
	}
}
