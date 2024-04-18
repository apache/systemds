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
import org.apache.sysds.runtime.functionobjects.MinusMultiply;
import org.apache.sysds.runtime.functionobjects.PlusMultiply;
import org.apache.sysds.runtime.functionobjects.TernaryValueFunction.ValueFunctionWithConstant;
import org.apache.sysds.runtime.matrix.data.LibMatrixTercell;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.TernaryOperator;

public final class CLALibTernaryOp {
	private CLALibTernaryOp() {
		// private constructor
	}

	public static MatrixBlock ternaryOperations(CompressedMatrixBlock m1, TernaryOperator op, MatrixBlock m2,
		MatrixBlock m3, MatrixBlock ret) {

		// prepare inputs
		final int r1 = m1.getNumRows();
		final int r2 = m2.getNumRows();
		final int r3 = m3.getNumRows();
		final int c1 = m1.getNumColumns();
		final int c2 = m2.getNumColumns();
		final int c3 = m3.getNumColumns();
		final boolean s1 = (r1 == 1 && c1 == 1);
		final boolean s2 = (r2 == 1 && c2 == 1);
		final boolean s3 = (r3 == 1 && c3 == 1);
		final double d1 = s1 ? m1.get(0, 0) : Double.NaN;
		final double d2 = s2 ? m2.get(0, 0) : Double.NaN;
		final double d3 = s3 ? m3.get(0, 0) : Double.NaN;
		final int m = Math.max(Math.max(r1, r2), r3);
		final int n = Math.max(Math.max(c1, c2), c3);

		MatrixBlock.ternaryOperationCheck(s1, s2, s3, m, r1, r2, r3, n, c1, c2, c3);

		final boolean PM_Or_MM = (op.fn instanceof PlusMultiply || op.fn instanceof MinusMultiply);
		if(PM_Or_MM && ((s2 && d2 == 0) || (s3 && d3 == 0))) {
			ret = new CompressedMatrixBlock();
			ret.copy(m1);
			return ret;
		}

		if(m2 instanceof CompressedMatrixBlock)
			m2 = ((CompressedMatrixBlock) m2).getUncompressed("Ternary Operator arg2 " + op.fn.getClass().getSimpleName(),
				op.getNumThreads());
		if(m3 instanceof CompressedMatrixBlock)
			m3 = ((CompressedMatrixBlock) m3).getUncompressed("Ternary Operator arg3 " + op.fn.getClass().getSimpleName(),
				op.getNumThreads());

		if(s2 != s3 && (op.fn instanceof PlusMultiply || op.fn instanceof MinusMultiply)) {
			// SPECIAL CASE for sparse-dense combinations of common +* and -*
			BinaryOperator bop = ((ValueFunctionWithConstant) op.fn).setOp2Constant(s2 ? d2 : d3);
			bop.setNumThreads(op.getNumThreads());
			ret = CLALibBinaryCellOp.binaryOperationsRight(bop, m1, s2 ? m3 : m2, ret);
		}
		else {
			final boolean sparseOutput = MatrixBlock.evalSparseFormatInMemory(m, n, (s1 ? m * n * (d1 != 0 ? 1 : 0) : m1.getNonZeros()) +
				Math.min(s2 ? m * n : m2.getNonZeros(), s3 ? m * n : m3.getNonZeros()));
			ret.reset(m, n, sparseOutput);
			final MatrixBlock thisUncompressed = m1.getUncompressed("Ternary Operation not supported");
			LibMatrixTercell.tercellOp(thisUncompressed, m2, m3, ret, op);
			ret.examSparsity();
		}
		return ret;
	}

}
