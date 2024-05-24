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

package org.apache.sysds.api.ropt.cost;

import org.apache.sysds.common.Types;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.controlprogram.caching.FrameObject;
import org.apache.sysds.runtime.controlprogram.caching.TensorObject;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse;
import org.apache.sysds.runtime.data.BasicTensorBlock;
import org.apache.sysds.runtime.data.TensorBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.meta.TensorCharacteristics;

public class VarStats 
{
	DataCharacteristics _dc;
	boolean _inmem = false;

	Types.DataType _dt = null;
	
//	public VarStats( long rlen, long clen, int blen, long nnz, boolean inmem ) {
//		_dc = new MatrixCharacteristics(rlen, clen, blen, nnz);
//		_inmem = inmem;
//	}

	public VarStats(DataCharacteristics dc, boolean inmem, Types.DataType dataType) {
		_dc = dc;
		_inmem = inmem;
		_dt = dataType;
	}

	public DataCharacteristics getDC() {
		return _dc;
	}

	public Types.DataType getDataType() { return _dt; }

	public long getM() {
		return _dc.getRows();
	}

	public long getN() {
		return _dc.getCols();
	}

	public double getS() {
		return !_dc.nnzKnown() ? 1.0 : getSparsity();
	}
	public double getSparsity() {
		return OptimizerUtils.getSparsity(_dc);
	}

	public long getCells() {
		if (_dc instanceof TensorCharacteristics) {
			long result = 1;
			for (int i = 0; i < _dc.getNumDims(); i++) {
				result *= _dc.getDim(i);
			}
			return result;
		} else {
			return _dc.getRows() * _dc.getCols();
		}
	}

	public double getCellsWithSparsity() {
		if (isSparse())
			return getSparsity() * getS();
		return (double) getCells();
	}

	public boolean isSparse() {
		if (_dc instanceof TensorCharacteristics) {
			// TODO: consider if sparse tensors are handled differently
			return false;
		} else {
			return MatrixBlock.evalSparseFormatInMemory(_dc);
		}
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("VarStats: ");
		if (_dt == Types.DataType.MATRIX) {
			sb.append("Matrix [");
			sb.append("rlen = ");
			sb.append(_dc.getRows());
			sb.append(", clen = ");
			sb.append(_dc.getCols());
			sb.append(", nnz = ");
			sb.append(_dc.getNonZeros());
		} else if (_dt == Types.DataType.FRAME) {
			sb.append("Frame [");
			sb.append("rlen = ");
			sb.append(_dc.getRows());
			sb.append(", clen = ");
			sb.append(_dc.getCols());
			sb.append(", nnz = ");
			sb.append(_dc.getNonZeros());
		} else if (_dt == Types.DataType.TENSOR) {
			sb.append("Tensor [dims=[");
			for (int i = 0; i < _dc.getNumDims(); i++) {
				sb.append(_dc.getDim(i));
				sb.append(" ");
			}
			sb.replace(sb.length() - 1, sb.length(), "]");
			sb.append(", nnz = ");
			sb.append(_dc.getNonZeros());
		} else if (_dt == Types.DataType.SCALAR) {
			sb.append("Scalar");
			return sb.toString();
		}

		sb.append(", inmem = ");
		sb.append(_inmem);
		sb.append("]");
		return sb.toString();
	}
	
	@Override
	public Object clone() {
		return new VarStats(_dc, _inmem, _dt);
	}
}
