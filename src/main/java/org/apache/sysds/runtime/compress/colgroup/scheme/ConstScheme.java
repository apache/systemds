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
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public class ConstScheme extends ACLAScheme {

	final double[] vals;
	boolean initialized = false;

	private ConstScheme(IColIndex cols, double[] vals, boolean initialized) {
		super(cols);
		this.vals = vals;
		this.initialized = initialized;
	}

	public static ICLAScheme create(ColGroupConst g) {
		return new ConstScheme(g.getColIndices(), g.getValues(), true);
	}

	public static ICLAScheme create(IColIndex cols, double[] vals) {
		return new ConstScheme(cols, vals, false);
	}

	@Override
	protected IColIndex getColIndices() {
		return cols;
	}

	@Override
	public ICLAScheme update(MatrixBlock data, IColIndex columns) {
		final int nRow = data.getNumRows();
		final int nColScheme = vals.length;
		for(int r = 0; r < nRow; r++)
			for(int c = 0; c < nColScheme; c++) {
				final double v = data.quickGetValue(r, cols.get(c));
				if(Double.compare(v, vals[c]) != 0)
					return updateToDDC(data, columns);
			}
		return this;
	}

	private ICLAScheme updateToDDC(MatrixBlock data, IColIndex columns) {
		return SchemeFactory.create(columns, CompressionType.DDC).update(data, columns);
	}

	@Override
	public AColGroup encode(MatrixBlock data, IColIndex columns) {
		validate(data, columns);
		// we assume that it is always valid.
		return ColGroupConst.create(columns, vals);
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
