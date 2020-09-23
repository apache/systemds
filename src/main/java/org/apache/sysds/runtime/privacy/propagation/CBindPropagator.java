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

package org.apache.sysds.runtime.privacy.propagation;

import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.privacy.PrivacyConstraint;
import org.apache.sysds.runtime.privacy.finegrained.DataRange;
import org.apache.sysds.runtime.privacy.finegrained.FineGrainedPrivacy;

public class CBindPropagator extends AppendPropagator {

	public CBindPropagator(MatrixBlock input1, PrivacyConstraint privacyConstraint1, MatrixBlock input2,
		PrivacyConstraint privacyConstraint2) {
		super(input1, privacyConstraint1, input2, privacyConstraint2);
	}

	@Override
	protected void appendInput2(FineGrainedPrivacy mergedConstraint, DataRange range,
		PrivacyConstraint.PrivacyLevel privacyLevel) {
		long rowBegin = range.getBeginDims()[0]; //same as before
		long colBegin = range.getBeginDims()[1] + input1.getNumColumns();
		long[] beginDims = new long[]{rowBegin, colBegin};

		long rowEnd = range.getEndDims()[0]; //same as before
		long colEnd = range.getEndDims()[1] + input1.getNumColumns();
		long[] endDims = new long[]{rowEnd, colEnd};

		DataRange outputRange = new DataRange(beginDims, endDims);

		mergedConstraint.put(outputRange, privacyLevel);
	}
}
