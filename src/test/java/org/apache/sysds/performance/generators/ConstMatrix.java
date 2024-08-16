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

package org.apache.sysds.performance.generators;

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.functionobjects.ReduceAll;
import org.apache.sysds.runtime.instructions.cp.AggregateUnaryCPInstruction.AUType;
import org.apache.sysds.runtime.matrix.data.LibMatrixCountDistinct;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.CountDistinctOperator;
import org.apache.sysds.test.TestUtils;

public class ConstMatrix implements Const<MatrixBlock> {

	protected MatrixBlock mb;
	protected final int nVal;

	public ConstMatrix(MatrixBlock mb) {
		this.mb = mb;
		this.nVal = (int) LibMatrixCountDistinct
			.estimateDistinctValues(mb,
				new CountDistinctOperator(AUType.COUNT_DISTINCT, Types.Direction.RowCol, ReduceAll.getReduceAllFnObject()))
			.get(0, 0);
	}

	public ConstMatrix(MatrixBlock mb, int nVal) {
		this.mb = mb;
		this.nVal = nVal;
	}

	public ConstMatrix(int r, int c, int nVal, double s) {
		this.mb = TestUtils.ceil(TestUtils.generateTestMatrixBlock(r, c, 0, nVal, s, 42));
		this.nVal = nVal;
	}

	@Override
	public MatrixBlock take() {
		return mb;
	}

	@Override
	public void generate(int N) throws InterruptedException {
		// do nothing
	}

	@Override
	public final boolean isEmpty() {
		return false;
	}

	@Override
	public final int defaultWaitTime() {
		return 0;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(this.getClass().getSimpleName());
		sb.append(" ( Rows:").append(mb.getNumRows());
		sb.append(", Cols:").append(mb.getNumColumns());
		sb.append(", Spar:").append(mb.getSparsity());
		sb.append(", Unique: ").append(nVal);
		sb.append(")");
		return sb.toString();
	}

	@Override
	public void change(MatrixBlock t) {
		mb = t;
	}
}
