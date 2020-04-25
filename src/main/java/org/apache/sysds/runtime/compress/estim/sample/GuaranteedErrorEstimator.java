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

package org.apache.sysds.runtime.compress.estim.sample;

import java.util.HashMap;

import org.apache.sysds.runtime.compress.ReaderColumnSelection;
import org.apache.sysds.runtime.compress.utils.DblArray;

public class GuaranteedErrorEstimator {

	/**
	 * M. Charikar, S. Chaudhuri, R. Motwani, and V. R. Narasayya, Towards estimation error guarantees for distinct
	 * values, PODS'00.
	 * 
	 * @param nRows            number of rows
	 * @param sampleSize       sample size
	 * @param sampleRowsReader a reader for the sampled rows
	 * @return error estimator
	 */
	@SuppressWarnings("unused")
	private static int guaranteedErrorEstimator(int nRows, int sampleSize, ReaderColumnSelection sampleRowsReader) {
		HashMap<DblArray, Integer> valsCount = getValCounts(sampleRowsReader);
		// number of values that occur only once
		int singltonValsCount = 0;
		int otherValsCount = 0;
		for(Integer c : valsCount.values()) {
			if(c == 1)
				singltonValsCount++;
			else
				otherValsCount++;
		}
		return (int) Math.round(otherValsCount + singltonValsCount * Math.sqrt(((double) nRows) / sampleSize));
	}

	private static HashMap<DblArray, Integer> getValCounts(ReaderColumnSelection sampleRowsReader) {
		HashMap<DblArray, Integer> valsCount = new HashMap<>();
		DblArray val = null;
		Integer cnt;
		while(null != (val = sampleRowsReader.nextRow())) {
			cnt = valsCount.get(val);
			if(cnt == null)
				cnt = 0;
			cnt++;
			valsCount.put(new DblArray(val), cnt);
		}
		return valsCount;
	}
}
