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

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.federated.FederatedData;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRange;
import org.apache.sysds.runtime.controlprogram.federated.FederationMap;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

public class KeepDataOnWorkerFederatedScheme extends DataPartitionFederatedScheme {
	@Override
	public Result doPartitioning(MatrixObject features, MatrixObject labels) {
		if (features.isFederated(FederationMap.FType.ROW)
				|| labels.isFederated(FederationMap.FType.ROW)) {

			// partition features
			List<MatrixObject> pFeatures = new ArrayList<MatrixObject>();
			FederationMap fedMapFeatures = features.getFedMapping();
			for(FederatedRange r : fedMapFeatures.getFederatedRanges()) {
				// TODO: This slicing is really ugly, rework
				MatrixObject slice = new MatrixObject(features);
				HashMap<FederatedRange, FederatedData> newFedHashMap = new HashMap<>();
				newFedHashMap.put(r, fedMapFeatures.getFederatedDataObject(r));
				slice.setFedMapping(new FederationMap(newFedHashMap));
				slice.getFedMapping().setType(FederationMap.FType.ROW);
				pFeatures.add(slice);
			}

			// partition labels
			List<MatrixObject> pLabels = new ArrayList<MatrixObject>();
			FederationMap fedMapLabels = features.getFedMapping();
			for(FederatedRange r : labels.getFedMapping().getFederatedRanges()) {
				// TODO: This slicing is really ugly, rework
				MatrixObject slice = new MatrixObject(labels);
				HashMap<FederatedRange, FederatedData> newFedHashMap = new HashMap<>();
				newFedHashMap.put(r, fedMapLabels.getFederatedDataObject(r));
				slice.setFedMapping(new FederationMap(newFedHashMap));
				slice.getFedMapping().setType(FederationMap.FType.ROW);
				pLabels.add(slice);
			}

			return new Result(pFeatures, pLabels);
		}
		else {
			throw new DMLRuntimeException(String.format("Paramserv func: " +
					"currently only supports row federated data"));
		}
	}
}
