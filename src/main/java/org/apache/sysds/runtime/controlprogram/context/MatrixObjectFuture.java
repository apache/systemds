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

package org.apache.sysds.runtime.controlprogram.context;

import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

import java.util.concurrent.Future;

public class MatrixObjectFuture extends MatrixObject
{
	protected Future<MatrixBlock> _futureData;

	public MatrixObjectFuture(ValueType vt, String file, Future<MatrixBlock> fmb) {
		super(vt, file, null);
		_futureData = fmb;
	}

	public MatrixObjectFuture(MatrixObject mo, Future<MatrixBlock> fmb) {
		super(mo.getValueType(), mo.getFileName(), mo.getMetaData());
		_futureData = fmb;
	}

	MatrixBlock getMatrixBlock() {
		try {
			return _futureData.get();
		}
		catch(Exception e) {
			throw new DMLRuntimeException(e);
		}
	}

	public MatrixBlock acquireRead() {
		return acquireReadIntern();
	}

	private synchronized MatrixBlock acquireReadIntern() {
		try {
			if(!isAvailableToRead())
				throw new DMLRuntimeException("MatrixObject not available to read.");
			if(_data != null)
				throw new DMLRuntimeException("_data must be null for future matrix object/block.");
			acquire(false, false);
			return _futureData.get();
		}

		catch(Exception e) {
			throw new DMLRuntimeException(e);
		}
	}

	public void release() {
		releaseIntern();
	}

	private synchronized void releaseIntern() {
		_futureData = null;
	}

	public synchronized void clearData(long tid) {
		_data = null;
		_futureData = null;
		clearCache();
		setCacheLineage(null);
		setDirty(false);
		setEmpty();
	}
}
