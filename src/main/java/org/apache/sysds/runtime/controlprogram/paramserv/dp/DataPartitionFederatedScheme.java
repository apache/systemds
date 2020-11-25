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

package org.apache.sysds.runtime.controlprogram.paramserv.dp;

import org.apache.sysds.common.Types;
import org.apache.sysds.lops.compile.Dag;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.federated.FederatedData;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRange;
import org.apache.sysds.runtime.controlprogram.federated.FederationMap;
import org.apache.sysds.runtime.controlprogram.paramserv.ParamservUtils;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.AggregateBinaryOperator;
import org.apache.sysds.runtime.matrix.operators.AggregateOperator;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.meta.MetaDataFormat;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

public abstract class DataPartitionFederatedScheme {

	public static final class Result {
		public final List<MatrixObject> pFeatures;
		public final List<MatrixObject> pLabels;
		public final int workerNum;

		public Result(List<MatrixObject> pFeatures, List<MatrixObject> pLabels, int workerNum) {
			this.pFeatures = pFeatures;
			this.pLabels = pLabels;
			this.workerNum = workerNum;
		}
	}

	public abstract Result doPartitioning(MatrixObject features, MatrixObject labels);

	/**
	 * Takes a row federated Matrix and slices it into a matrix for each worker
	 *
	 * @param fedMatrix the federated input matrix
	 */
	static List<MatrixObject> sliceFederatedMatrix(MatrixObject fedMatrix) {
		if (fedMatrix.isFederated(FederationMap.FType.ROW)) {
			List<MatrixObject> slices = Collections.synchronizedList(new ArrayList<>());
			fedMatrix.getFedMapping().forEachParallel((range, data) -> {
				// Create sliced matrix object
				MatrixObject slice = new MatrixObject(fedMatrix.getValueType(), Dag.getNextUniqueVarname(Types.DataType.MATRIX));
				slice.setMetaData(new MetaDataFormat(
						new MatrixCharacteristics(range.getSize(0), range.getSize(1)),
						Types.FileFormat.BINARY)
				);

				// Create new federation map
				HashMap<FederatedRange, FederatedData> newFedHashMap = new HashMap<>();
				newFedHashMap.put(range, data);
				slice.setFedMapping(new FederationMap(fedMatrix.getFedMapping().getID(), newFedHashMap));
				slice.getFedMapping().setType(FederationMap.FType.ROW);

				slices.add(slice);
				return null;
			});

			return slices;
		}
		else {
			throw new DMLRuntimeException("Federated data partitioner: " +
					"currently only supports row federated data");
		}
	}

	/**
	 * Takes a MatrixObjects and shuffles it
	 *
	 * @param m the input matrix object
	 */
	static void shuffle(MatrixObject m) {
		// generate permutation matrix
		MatrixBlock permutationMatrixBlock = ParamservUtils.generatePermutation(Math.toIntExact(m.getNumRows()), System.currentTimeMillis());
		// matrix multiplies
		m.acquireModify(permutationMatrixBlock.aggregateBinaryOperations(
				permutationMatrixBlock, m.acquireReadAndRelease(), new MatrixBlock(),
				new AggregateBinaryOperator(Multiply.getMultiplyFnObject(), new AggregateOperator(0, Plus.getPlusFnObject()))
		));
		m.release();
	}

	/**
	 * Takes a MatrixObjects and extends it to the chosen number of rows by random replication
	 *
	 * @param m the input matrix object
	 */
	static void replicateTo(MatrixObject m, int rows) {
		int num_rows_needed = rows - Math.toIntExact(m.getNumRows());
		// generate replication matrix
		MatrixBlock replicateMatrixBlock = ParamservUtils.generateReplicationMatrix(num_rows_needed, Math.toIntExact(m.getNumRows()), System.currentTimeMillis());
		// matrix multiplies and append
		MatrixBlock replicatedFeatures = replicateMatrixBlock.aggregateBinaryOperations(
				replicateMatrixBlock, m.acquireReadAndRelease(), new MatrixBlock(),
				new AggregateBinaryOperator(Multiply.getMultiplyFnObject(), new AggregateOperator(0, Plus.getPlusFnObject())));

		m.acquireModify(m.acquireReadAndRelease().append(replicatedFeatures, new MatrixBlock(), false));
		m.release();
	}

	/**
	 * Takes a MatrixObjects and shrinks it to the given number of rows by subsampling
	 *
	 * @param m the input matrix object
	 */
	static void subsampleTo(MatrixObject m, int rows) {
		// generate subsampling matrix
		MatrixBlock subsampleMatrixBlock = ParamservUtils.generateSubsampleMatrix(rows, Math.toIntExact(m.getNumRows()), System.currentTimeMillis());
		// matrix multiplies
		m.acquireModify(subsampleMatrixBlock.aggregateBinaryOperations(
				subsampleMatrixBlock, m.acquireReadAndRelease(), new MatrixBlock(),
				new AggregateBinaryOperator(Multiply.getMultiplyFnObject(), new AggregateOperator(0, Plus.getPlusFnObject()))
		));
		m.release();
	}
}
