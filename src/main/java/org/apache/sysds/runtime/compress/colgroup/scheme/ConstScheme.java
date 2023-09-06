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

package org.apache.sysds.runtime.compress.colgroup.scheme;

import java.util.Arrays;

import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.AColGroup.CompressionType;
import org.apache.sysds.runtime.compress.colgroup.ColGroupConst;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.utils.Util;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public class ConstScheme extends ACLAScheme {

	final double[] vals;

	private ConstScheme(IColIndex cols, double[] vals) {
		super(cols);
		this.vals = vals;
	}

	public static ICLAScheme create(ColGroupConst g) {
		return new ConstScheme(g.getColIndices(), g.getValues());
	}

	public static ICLAScheme create(IColIndex cols, double[] vals) {
		if(vals == null)
			throw new RuntimeException("Invalid null vals for ConstScheme");
		return new ConstScheme(cols, vals);
	}

	@Override
	protected ICLAScheme updateV(MatrixBlock data, IColIndex columns) {
		final int nRow = data.getNumRows();
		final int nColScheme = vals.length;
		for(int r = 0; r < nRow; r++)
			for(int c = 0; c < nColScheme; c++) {
				final double v = data.quickGetValue(r, cols.get(c));
				if(!Util.eq(v, vals[c]))
					return updateToDDC(data, columns);
			}
		return this;
	}

	private ICLAScheme updateToDDC(MatrixBlock data, IColIndex columns) {
		return SchemeFactory.create(columns, CompressionType.DDC).update(data, columns);
	}

	private ICLAScheme updateToDDCT(MatrixBlock data, IColIndex columns) {
		return SchemeFactory.create(columns, CompressionType.DDC).updateT(data, columns);
	}

	@Override
	protected AColGroup encodeV(MatrixBlock data, IColIndex columns) {
		return ColGroupConst.create(columns, vals);
	}

	@Override
	protected AColGroup encodeVT(MatrixBlock data, IColIndex columns) {
		return ColGroupConst.create(columns, vals);
	}

	@Override
	protected ICLAScheme updateVT(MatrixBlock data, IColIndex columns) {
		// TODO specialize for sparse data. But would only be used in rare cases
		final int nCol = data.getNumColumns();
		final int nColScheme = vals.length;
		for(int r = 0; r < nColScheme; r++) {
			final int row = cols.get(r);
			final double def = vals[r];
			for(int c = 0; c < nCol; c++) {
				final double v = data.quickGetValue(row, c);
				if(!Util.eq(v, def))
					return updateToDDCT(data, columns);
			}
		}
		return this;
	}

	@Override
	public ConstScheme clone() {
		return new ConstScheme(cols, Arrays.copyOf(vals, vals.length));
	}

	@Override
	public final String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(this.getClass().getSimpleName());
		sb.append(" Cols: ");
		sb.append(cols);
		sb.append(" Def:  ");
		sb.append(Arrays.toString(vals));
		return sb.toString();
	}

}
