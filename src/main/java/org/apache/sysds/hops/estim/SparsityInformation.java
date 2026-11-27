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

package org.apache.sysds.hops.estim;

import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;

public abstract class SparsityInformation {
	protected MatrixBlock _data = null;
	protected final int _rlen;
	protected final int _clen;

	public SparsityInformation(int rlen, int clen) {
		_rlen = rlen;
		_clen = clen;
	}

	public SparsityInformation(MatrixBlock mb) {
		_data = mb;
		_rlen = mb.getNumRows();
		_clen = mb.getNumColumns();
	}

	public void setData(MatrixBlock mb) {
		_data = mb;
	}

	public int getNumRows() {
		return _rlen;
	}

	public int getNumColumns() {
		return _clen;
	}

	public abstract long getNnz();

	public double getAvgSparsity() {
		// FIXME: do not return sparsity of 1 when nnz is unknown
		// return (double)getNonZeros() / (getNumRows()*getNumColumns());
		return OptimizerUtils.getSparsity(getNumRows(), getNumColumns(), getNnz());
	}

	public DataCharacteristics deriveDataCharacteristics() {
		return new MatrixCharacteristics(getNumRows(), getNumColumns(), getNnz());
	}
}
