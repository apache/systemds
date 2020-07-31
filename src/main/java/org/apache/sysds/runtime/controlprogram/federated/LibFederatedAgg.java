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

package org.apache.sysds.runtime.controlprogram.federated;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.functionobjects.KahanFunction;
import org.apache.sysds.runtime.functionobjects.ValueFunction;
import org.apache.sysds.runtime.instructions.cp.KahanObject;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Future;

/**
 * Library for federated aggregation operations.
 * <p>
 * This libary covers the following opcodes:
 * uak+
 */
public class LibFederatedAgg
{
	public static MatrixBlock aggregateUnaryMatrix(MatrixObject federatedMatrix, AggregateUnaryOperator operator) {
		// find out the characteristics after aggregation
		MatrixCharacteristics mc = new MatrixCharacteristics();
		operator.indexFn.computeDimension(federatedMatrix.getDataCharacteristics(), mc);
		// make outBlock right size
		MatrixBlock ret = new MatrixBlock((int) mc.getRows(), (int) mc.getCols(), operator.aggOp.initialValue);
		List<Pair<FederatedRange, Future<FederatedResponse>>> idResponsePairs = new ArrayList<>();
		// distribute aggregation operation to all workers
		for (Map.Entry<FederatedRange, FederatedData> entry : federatedMatrix.getFedMapping().entrySet()) {
			FederatedData fedData = entry.getValue();
			if (!fedData.isInitialized())
				throw new DMLRuntimeException("Not all FederatedData was initialized for federated matrix");
			Future<FederatedResponse> future = fedData.executeFederatedOperation(
				new FederatedRequest(FederatedRequest.FedMethod.AGGREGATE, operator), true);
			idResponsePairs.add(new ImmutablePair<>(entry.getKey(), future));
		}
		try {
			//TODO replace with block operations
			for (Pair<FederatedRange, Future<FederatedResponse>> idResponsePair : idResponsePairs) {
				FederatedRange range = idResponsePair.getLeft();
				FederatedResponse federatedResponse = idResponsePair.getRight().get();
				int[] beginDims = range.getBeginDimsInt();
				MatrixBlock mb = (MatrixBlock) federatedResponse.getData()[0];
				// TODO performance optimizations
				MatrixValue.CellIndex cellIndex = new MatrixValue.CellIndex(0, 0);
				ValueFunction valueFn = operator.aggOp.increOp.fn;
				// Add worker response to resultBlock
				for (int r = 0; r < mb.getNumRows(); r++)
					for (int c = 0; c < mb.getNumColumns(); c++) {
						// Get the output index where the result should be placed by the index function
						// -> map input row/col to output row/col
						cellIndex.set(r + beginDims[0], c + beginDims[1]);
						operator.indexFn.execute(cellIndex, cellIndex);
						int resultRow = cellIndex.row;
						int resultCol = cellIndex.column;
						double newValue;
						if (valueFn instanceof KahanFunction) {
							// TODO iterate along correct axis to use correction correctly
							// temporary solution to execute correct overloaded method
							KahanObject kobj = new KahanObject(ret.quickGetValue(resultRow, resultCol), 0);
							newValue = ((KahanObject) valueFn.execute(kobj, mb.quickGetValue(r, c)))._sum;
						}
						else {
							// TODO special handling for `ValueFunction`s which do not implement `.execute(double, double)`
							// "Add" two partial calculations together with ValueFunction
							newValue = valueFn.execute(ret.quickGetValue(resultRow, resultCol), mb.quickGetValue(r, c));
						}
						ret.quickSetValue(resultRow, resultCol, newValue);
					}
			}
		}
		catch (Exception e) {
			throw new DMLRuntimeException("Federated binary aggregation failed", e);
		}
		return ret;
	}
}
