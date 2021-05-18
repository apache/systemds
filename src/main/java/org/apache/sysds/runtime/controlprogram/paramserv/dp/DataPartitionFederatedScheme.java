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

import org.apache.commons.lang3.tuple.Pair;
import org.apache.sysds.common.Types;
import org.apache.sysds.lops.compile.Dag;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.federated.FederatedData;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRange;
import org.apache.sysds.runtime.controlprogram.federated.FederationMap;
import org.apache.sysds.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.AggregateBinaryOperator;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.meta.MetaDataFormat;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public abstract class DataPartitionFederatedScheme {

	public static final class Result {
		public final List<MatrixObject> _pFeatures;
		public final List<MatrixObject> _pLabels;
		public final int _workerNum;
		public final BalanceMetrics _balanceMetrics;
		public final List<Double> _weightingFactors;


		public Result(List<MatrixObject> pFeatures, List<MatrixObject> pLabels, int workerNum, BalanceMetrics balanceMetrics, List<Double> weightingFactors) {
			_pFeatures = pFeatures;
			_pLabels = pLabels;
			_workerNum = workerNum;
			_balanceMetrics = balanceMetrics;
			_weightingFactors = weightingFactors;
		}
	}

	public static final class BalanceMetrics {
		public final long _minRows;
		public final long _avgRows;
		public final long _maxRows;

		public BalanceMetrics(long minRows, long avgRows, long maxRows) {
			_minRows = minRows;
			_avgRows = avgRows;
			_maxRows = maxRows;
		}
	}

	public abstract Result partition(MatrixObject features, MatrixObject labels, int seed);

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
				List<Pair<FederatedRange, FederatedData>> newFedHashMap = new ArrayList<>();
				newFedHashMap.add(Pair.of(range, data));
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

	static BalanceMetrics getBalanceMetrics(List<MatrixObject> slices) {
		if (slices == null || slices.size() == 0)
			return new BalanceMetrics(0, 0, 0);

		long minRows = slices.get(0).getNumRows();
		long maxRows = minRows;
		long sum = 0;

		for (MatrixObject slice : slices) {
			if (slice.getNumRows() < minRows)
				minRows = slice.getNumRows();
			else if (slice.getNumRows() > maxRows)
				maxRows = slice.getNumRows();

			sum += slice.getNumRows();
		}

		return new BalanceMetrics(minRows, sum / slices.size(), maxRows);
	}

	static List<Double> getWeightingFactors(List<MatrixObject> pFeatures, BalanceMetrics balanceMetrics) {
		List<Double> weightingFactors = new ArrayList<>();
		pFeatures.forEach((feature) -> {
			weightingFactors.add((double) feature.getNumRows() / balanceMetrics._avgRows);
		});
		return weightingFactors;
	}

	/**
	 * Just a mat multiply used to shuffle with a provided shuffle matrixBlock
	 *
	 * @param m the input matrix object
	 * @param P the permutation matrix for shuffling
	 */
	static void shuffle(MatrixObject m, MatrixBlock P) {
		int k = InfrastructureAnalyzer.getLocalParallelism();
		AggregateBinaryOperator mm = InstructionUtils.getMatMultOperator(k);
		MatrixBlock out = P.aggregateBinaryOperations(P, m.acquireReadAndRelease(), new MatrixBlock(), mm);
		m.acquireModify(out);
		m.release();
	}

	/**
	 * Takes a MatrixObjects and extends it to the chosen number of rows by random replication
	 *
	 * @param m the input matrix object
	 * @param R the permutation matrix for replication
	 */
	static void replicateTo(MatrixObject m, MatrixBlock R) {
		int k = InfrastructureAnalyzer.getLocalParallelism();
		AggregateBinaryOperator mm = InstructionUtils.getMatMultOperator(k);
		MatrixBlock out = R.aggregateBinaryOperations(R, m.acquireReadAndRelease(), new MatrixBlock(), mm);
		m.acquireModify(m.acquireReadAndRelease().append(out, new MatrixBlock(), false));
		m.release();
	}

	/**
	 * Just a mat multiply used to subsample with a provided subsample matrixBlock
	 *
	 * @param m the input matrix object
	 * @param R the permutation matrix for subsampling
	 */
	static void subsampleTo(MatrixObject m, MatrixBlock S) {
		int k = InfrastructureAnalyzer.getLocalParallelism();
		AggregateBinaryOperator mm = InstructionUtils.getMatMultOperator(k);
		MatrixBlock out = S.aggregateBinaryOperations(S, m.acquireReadAndRelease(), new MatrixBlock(), mm);
		m.acquireModify(out);
		m.release();
	}
}
