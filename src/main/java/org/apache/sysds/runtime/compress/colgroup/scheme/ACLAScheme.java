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

import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.Pair;

public abstract class ACLAScheme implements ICLAScheme {
	protected final IColIndex cols;

	protected ACLAScheme(IColIndex cols) {
		this.cols = cols;
	}

	@Override
	public final AColGroup encode(MatrixBlock data) {
		return encode(data, cols);
	}

	@Override
	public final AColGroup encode(MatrixBlock data, IColIndex columns) {
		validate(data, columns);
		return encodeV(data, columns);
	}

	protected abstract AColGroup encodeV(MatrixBlock data, IColIndex columns);

	@Override
	public final AColGroup encodeT(MatrixBlock data) {
		return encodeVT(data, cols);
	}

	@Override
	public final AColGroup encodeT(MatrixBlock data, IColIndex columns) {
		validateT(data, columns);
		return encodeVT(data, columns);
	}

	protected abstract AColGroup encodeVT(MatrixBlock data, IColIndex columns);

	@Override
	public final ICLAScheme update(MatrixBlock data) {
		return updateV(data, cols);
	}

	@Override
	public final ICLAScheme update(MatrixBlock data, IColIndex columns) {
		validate(data, columns);
		return updateV(data, columns);
	}

	protected abstract ICLAScheme updateV(MatrixBlock data, IColIndex columns);

	@Override
	public final ICLAScheme updateT(MatrixBlock data) {
		return updateVT(data, cols);
	}

	@Override
	public final ICLAScheme updateT(MatrixBlock data, IColIndex columns) {
		validateT(data, columns);
		return updateVT(data, columns);
	}

	protected abstract ICLAScheme updateVT(MatrixBlock data, IColIndex columns);

	@Override
	public final Pair<ICLAScheme, AColGroup> updateAndEncode(MatrixBlock data) {
		return updateAndEncode(data, cols);
	}

	@Override
	public final Pair<ICLAScheme, AColGroup> updateAndEncodeT(MatrixBlock data) {
		return updateAndEncodeT(data, cols);
	}

	@Override
	public final Pair<ICLAScheme, AColGroup> updateAndEncode(MatrixBlock data, IColIndex columns) {
		validate(data, columns);
		try {
			return tryUpdateAndEncode(data, columns);
		}
		catch(Exception e) {
			return fallBackUpdateAndEncode(data, columns);
		}
	}

	@Override
	public final Pair<ICLAScheme, AColGroup> updateAndEncodeT(MatrixBlock data, IColIndex columns) {
		validateT(data, columns);
		try {
			return tryUpdateAndEncodeT(data, columns);
		}
		catch(Exception e) {
			return fallBackUpdateAndEncodeT(data, columns);
		}
	}

	protected Pair<ICLAScheme, AColGroup> tryUpdateAndEncode(MatrixBlock data, IColIndex columns) {
		return fallBackUpdateAndEncode(data, columns);
	}

	protected Pair<ICLAScheme, AColGroup> tryUpdateAndEncodeT(MatrixBlock data, IColIndex columns) {
		return fallBackUpdateAndEncodeT(data, columns);
	}

	private final Pair<ICLAScheme, AColGroup> fallBackUpdateAndEncode(MatrixBlock data, IColIndex columns) {
		final ICLAScheme s = update(data, columns);
		final AColGroup g = s.encode(data, columns);
		return new Pair<>(s, g);
	}

	private final Pair<ICLAScheme, AColGroup> fallBackUpdateAndEncodeT(MatrixBlock data, IColIndex columns) {
		final ICLAScheme s = updateT(data, columns);
		final AColGroup g = s.encodeT(data, columns);
		return new Pair<>(s, g);
	}

	private final void validate(MatrixBlock data, IColIndex columns) throws IllegalArgumentException {
		if(columns.size() != cols.size())
			throw new IllegalArgumentException(
				"Invalid number of columns to encode expected: " + cols.size() + " but got: " + columns.size());

		final int nCol = data.getNumColumns();
		if(nCol < cols.get(cols.size() - 1))
			throw new IllegalArgumentException(
				"Invalid columns to encode with max col:" + nCol + " list of columns: " + columns);
	}

	private final void validateT(MatrixBlock data, IColIndex columns) throws IllegalArgumentException {
		if(columns.size() != cols.size())
			throw new IllegalArgumentException(
				"Invalid number of columns to encode expected: " + cols.size() + " but got: " + columns.size());

		final int nRow = data.getNumRows();
		if(nRow < cols.get(cols.size() - 1))
			throw new IllegalArgumentException(
				"Invalid columns to encode with max col:" + nRow + " list of columns: " + columns);
	}

	@Override
	public abstract ACLAScheme clone();

}
