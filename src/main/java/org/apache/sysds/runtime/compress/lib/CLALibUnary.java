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

import java.util.ArrayList;
import java.util.List;

import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Builtin.BuiltinCode;
import org.apache.sysds.runtime.matrix.data.LibMatrixAgg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;

public class CLALibUnary {
	// private static final Log LOG = LogFactory.getLog(CLALibUnary.class.getName());

	public static MatrixBlock unaryOperations(CompressedMatrixBlock m, UnaryOperator op, MatrixValue result) {
		final boolean overlapping = m.isOverlapping();
		final int r = m.getNumRows();
		final int c = m.getNumColumns();
		// early aborts:
		if(m.isEmpty())
			return new MatrixBlock(r, c, 0).unaryOperations(op, result);
		else if(overlapping) {
			// when in overlapping state it is guaranteed that there is no infinites, NA, or NANs.
			if(Builtin.isBuiltinCode(op.fn, BuiltinCode.ISINF, BuiltinCode.ISNA, BuiltinCode.ISNAN))
				return new MatrixBlock(r, c, 0);
			if(op.fn instanceof Builtin)
			return m.getUncompressed("Unary Op not supported Overlapping builtin: " + ((Builtin)(op.fn)).getBuiltinCode(), op.getNumThreads()).unaryOperations(op, null);
			else
				return m.getUncompressed("Unary Op not supported Overlapping: " + op.fn.getClass().getSimpleName(), op.getNumThreads()).unaryOperations(op, null);
		}
		else if(Builtin.isBuiltinCode(op.fn, BuiltinCode.ISINF, BuiltinCode.ISNAN, BuiltinCode.ISNA) &&
			!m.containsValue(op.getPattern()))
			return new MatrixBlock(r, c, 0); // avoid unnecessary allocation
		else if(LibMatrixAgg.isSupportedUnaryOperator(op)) {
			// e.g., cumsum/cumprod/cummin/cumax/cumsumprod
			return m.getUncompressed("Unary Op not supported: " + op.fn.getClass().getSimpleName(), op.getNumThreads()).unaryOperations(op, null);
		}
		else {

			List<AColGroup> groups = m.getColGroups();
			List<AColGroup> retG = new ArrayList<>(groups.size());
			for(AColGroup g : groups)
				retG.add(g.unaryOperation(op));

			CompressedMatrixBlock ret = new CompressedMatrixBlock(m.getNumRows(), m.getNumColumns());
			ret.allocateColGroupList(retG);
			ret.recomputeNonZeros();
			return ret;
		}

	}
}
