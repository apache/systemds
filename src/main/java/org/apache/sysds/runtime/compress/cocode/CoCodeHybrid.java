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

package org.apache.sysds.runtime.compress.cocode;

import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.cost.ICostEstimate;
import org.apache.sysds.runtime.compress.estim.CompressedSizeEstimator;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfo;
import org.apache.sysds.runtime.controlprogram.parfor.stat.Timing;

/**
 * This cocode strategy starts combines the use of CoCodePriorityQue and CoCodeGreedy.
 */
public class CoCodeHybrid extends AColumnCoCoder {

	protected CoCodeHybrid(CompressedSizeEstimator sizeEstimator, ICostEstimate costEstimator, CompressionSettings cs) {
		super(sizeEstimator, costEstimator, cs);
	}

	@Override
	protected CompressedSizeInfo coCodeColumns(CompressedSizeInfo colInfos, int k) {
		final int startSize = colInfos.getInfo().size();
		final int PriorityQueGoal = startSize / 5;
		LOG.debug("Using Hybrid Cocode Strategy: ");
		if(startSize == 1)
			return colInfos;
		else if(startSize > 1000) // Large number of columns, then we only use priority que.
			colInfos.setInfo(CoCodePriorityQue.join(colInfos.getInfo(), _sest, _cest, 1));
		else if(startSize <= 5) // Greedy all compare all if small number of columns
			colInfos.setInfo(CoCodeGreedy.join(colInfos.getInfo(), _sest, _cest, _cs, k));
		else if(PriorityQueGoal > 30) { // hybrid if there is a large number of columns to begin with
			Timing time = new Timing(true);
			colInfos.setInfo(CoCodePriorityQue.join(colInfos.getInfo(), _sest, _cest, PriorityQueGoal));
			LOG.debug("Que based time: " + time.stop());
			final int pqSize = colInfos.getInfo().size();
			if(pqSize <= PriorityQueGoal * 2) {
				time = new Timing(true);
				colInfos.setInfo(CoCodeGreedy.join(colInfos.getInfo(), _sest, _cest, _cs, k));
				LOG.debug("Greedy time:     " + time.stop());
			}
		}
		else // If somewhere in between use the que based approach only.
			colInfos.setInfo(CoCodePriorityQue.join(colInfos.getInfo(), _sest, _cest, 1));

		return colInfos;
	}
}
