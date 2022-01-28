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

import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.UtilFunctions;

public class CLALibRexpand {

	// private static final Log LOG = LogFactory.getLog(CLALibReExpand.class.getName());

	public static MatrixBlock rexpand(CompressedMatrixBlock in, MatrixBlock ret, double max, boolean rows, boolean cast,
		boolean ignore, int k) {
		if(rows)
			return in.getUncompressed("Rexpand in rows direction (one hot encode)").rexpandOperations(ret, max, rows, cast,
				ignore, k);
		else
			return rexpandCols(in, max, cast, ignore, k);
	}

	private static MatrixBlock rexpandCols(CompressedMatrixBlock in, double max, boolean cast, boolean ignore, int k) {
		return rexpandCols(in, UtilFunctions.toInt(max), cast, ignore, k);
	}

	private static MatrixBlock rexpandCols(CompressedMatrixBlock in, int max, boolean cast, boolean ignore, int k) {
		LibMatrixReorg.checkRexpand(in, ignore);

		final int nRows = in.getNumRows();
		if(in.isEmptyBlock(false))
			return new MatrixBlock(nRows, max, true);
		else if(in.isOverlapping() || in.getColGroups().size() > 1)
			return LibMatrixReorg.rexpand(in.getUncompressed("Rexpand (one hot encode)"), new MatrixBlock(), max, false,
				cast, ignore, k);
		else {
			CompressedMatrixBlock retC = new CompressedMatrixBlock(nRows, max);
			retC.allocateColGroup(in.getColGroups().get(0).rexpandCols(max, ignore, cast, nRows));
			retC.recomputeNonZeros();
			return retC;
		}
	}
}
