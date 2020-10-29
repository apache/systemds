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
				// Warning needs MetaDataFormat instead of MetaData
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
}
