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

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.CacheStatistics;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.caching.UnifiedMemoryManager;
import org.apache.sysds.runtime.lineage.LineageCache;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.utils.stats.SparkStatistics;

import java.util.concurrent.Future;

public class MatrixObjectFuture extends MatrixObject
{
	private static final long serialVersionUID = 8016955240352642173L;
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
		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		//core internal acquire (synchronized per object)
		MatrixBlock ret = acquireReadIntern();

		if( DMLScript.STATISTICS ){
			long t1 = System.nanoTime();
			CacheStatistics.incrementAcquireRTime(t1-t0);
			SparkStatistics.incAsyncSparkOpCount(1);
		}
		return ret;
	}

	private synchronized MatrixBlock acquireReadIntern() {
		try {
			if(!isAvailableToRead())
				throw new DMLRuntimeException("MatrixObject not available to read.");
			if (_futureData == null)
				return super.acquireRead(); //moved to _data

			if (OptimizerUtils.isUMMEnabled())
				//track and make space in the UMM
				UnifiedMemoryManager.pin(this);

			MatrixBlock out = null;
			long t1 = System.nanoTime();
			out = _futureData.get();
			if (hasValidLineage())
				LineageCache.putValueAsyncOp(getCacheLineage(), this, out, t1);
				// FIXME: start time should indicate the actual start of the execution
			return acquireModify(out);
		}
		catch(Exception e) {
			throw new DMLRuntimeException(e);
		}
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
