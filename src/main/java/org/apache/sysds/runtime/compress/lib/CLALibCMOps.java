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

package org.apache.sysds.runtime.compress.lib;

import java.util.List;

import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.instructions.cp.CmCovObject;
import org.apache.sysds.runtime.matrix.data.LibMatrixAgg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.CMOperator;

public final class CLALibCMOps {

	private CLALibCMOps() {
		// private constructor
	}

	public static CmCovObject centralMoment(CompressedMatrixBlock cmb, CMOperator op) {
		MatrixBlock.checkCMOperations(cmb, op);
		if(cmb.isEmpty())
			return LibMatrixAgg.aggregateCmCov(cmb, null, null, op.fn);
		else if(cmb.isOverlapping())
			return cmb.getUncompressed("cmOperations on overlapping state", op.getNumThreads()).cmOperations(op);

		final List<AColGroup> groups = cmb.getColGroups();
		if(groups.size() == 1)
			return groups.get(0).centralMoment(op, cmb.getNumRows());

		return cmb.getUncompressed(
			"Decompressing but should never happen that a single column is non overlapping and contain multiple groups",
			op.getNumThreads()).cmOperations(op);

	}
}
