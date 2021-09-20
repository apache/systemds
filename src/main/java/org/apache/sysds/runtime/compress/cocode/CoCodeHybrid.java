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

/**
 * This cocode strategy starts out with priority que until a threshold number of columnGroups is achieved, then the
 * strategy shifts into a greedy all compare.
 */
public class CoCodeHybrid extends AColumnCoCoder {

    protected CoCodeHybrid(CompressedSizeEstimator sizeEstimator, ICostEstimate costEstimator, CompressionSettings cs) {
        super(sizeEstimator, costEstimator, cs);
    }

    @Override
    protected CompressedSizeInfo coCodeColumns(CompressedSizeInfo colInfos, int k) {
        final int startSize = colInfos.getInfo().size();
        final int PriorityQueGoal = 40;
        if(startSize > 200) {

            colInfos.setInfo(CoCodePriorityQue.join(colInfos.getInfo(), _sest, _cest, PriorityQueGoal));

            final int pqSize = colInfos.getInfo().size();
            if(pqSize <= PriorityQueGoal)
                colInfos.setInfo(CoCodeGreedy.join(colInfos.getInfo(), _sest, _cest, _cs));
        }
        else {
            colInfos.setInfo(CoCodeGreedy.join(colInfos.getInfo(), _sest, _cest, _cs));
        }

        return colInfos;
    }

}
