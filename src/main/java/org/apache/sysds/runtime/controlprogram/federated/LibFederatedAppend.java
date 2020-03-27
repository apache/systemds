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

import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.meta.DataCharacteristics;

import java.util.Map;
import java.util.TreeMap;

public class LibFederatedAppend {
	public static MatrixObject federateAppend(MatrixObject matObject1, MatrixObject matObject2,
		MatrixObject matObjectRet, boolean cbind)
	{
		Map<FederatedRange, FederatedData> fedMapping = new TreeMap<>();
		DataCharacteristics dc = matObjectRet.getDataCharacteristics();
		if (cbind) {
			// check for same amount of rows for matObject1 and matObject2 should have been checked before call
			dc.setRows(matObject1.getNumRows());
			// added because cbind
			long columnsLeftMat = matObject1.getNumColumns();
			dc.setCols(columnsLeftMat + matObject2.getNumColumns());
			
			Map<FederatedRange, FederatedData> fedMappingLeft = matObject1.getFedMapping();
			for (Map.Entry<FederatedRange, FederatedData> entry : fedMappingLeft.entrySet()) {
				// note that FederatedData should not change its varId once set
				fedMapping.put(new FederatedRange(entry.getKey()), entry.getValue());
			}
			Map<FederatedRange, FederatedData> fedMappingRight = matObject2.getFedMapping();
			for (Map.Entry<FederatedRange, FederatedData> entry : fedMappingRight.entrySet()) {
				// add offset due to cbind
				FederatedRange range = new FederatedRange(entry.getKey());
				range.setBeginDim(1, columnsLeftMat + range.getBeginDims()[1]);
				range.setEndDim(1, columnsLeftMat + range.getEndDims()[1]);
				fedMapping.put(range, entry.getValue());
			}
		}
		else {
			// check for same amount of cols for matObject1 and matObject2 should have been checked before call
			dc.setCols(matObject1.getNumColumns());
			// added because rbind
			long rowsUpperMat = matObject1.getNumRows();
			dc.setRows(rowsUpperMat + matObject2.getNumRows());
			
			Map<FederatedRange, FederatedData> fedMappingUpper = matObject1.getFedMapping();
			for (Map.Entry<FederatedRange, FederatedData> entry : fedMappingUpper.entrySet()) {
				// note that FederatedData should not change its varId once set
				fedMapping.put(new FederatedRange(entry.getKey()), entry.getValue());
			}
			Map<FederatedRange, FederatedData> fedMappingLower = matObject2.getFedMapping();
			for (Map.Entry<FederatedRange, FederatedData> entry : fedMappingLower.entrySet()) {
				// add offset due to rbind
				FederatedRange range = new FederatedRange(entry.getKey());
				range.setBeginDim(0, rowsUpperMat + range.getBeginDims()[0]);
				range.setEndDim(0, rowsUpperMat + range.getEndDims()[0]);
				fedMapping.put(range, entry.getValue());
			}
		}
		matObjectRet.setFedMapping(fedMapping);
		dc.setNonZeros(matObject1.getNnz() + matObject2.getNnz());
		return matObjectRet;
	}
}
