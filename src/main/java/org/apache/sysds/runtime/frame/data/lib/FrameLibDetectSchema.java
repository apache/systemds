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

package org.apache.sysds.runtime.frame.data.lib;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.estim.ComEstFactory;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.columns.Array;
import org.apache.sysds.runtime.matrix.data.Pair;
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.apache.sysds.runtime.util.UtilFunctions;

public final class FrameLibDetectSchema {
	protected static final Log LOG = LogFactory.getLog(FrameLibDetectSchema.class.getName());
	/** Default minium sample size */
	private static final int DEFAULT_MIN_CELLS = 10000;
	private static final int DEFAULT_MAX_CELLS = 1000000;

	/** Frame block to sample from */
	private final FrameBlock in;
	/** parallelization degree */
	private final int k;
	/** Sample size in case above nCells */
	private final int sampleSize;

	private FrameLibDetectSchema(FrameBlock in, double sampleFraction, int k) {
		this.in = in;
		this.k = k;
		final int inRows = in.getNumRows();
		this.sampleSize = Math.min(Math.max((int) (inRows * sampleFraction), DEFAULT_MIN_CELLS),
			ComEstFactory.getSampleSize(0.65, inRows, in.getNumColumns(), 1.0, DEFAULT_MIN_CELLS, DEFAULT_MAX_CELLS));
	}

	public static FrameBlock detectSchema(FrameBlock in, int k) {
		return new FrameLibDetectSchema(in, 1.0, k).apply();
	}

	public static FrameBlock detectSchema(FrameBlock in, double sampleFraction, int k) {
		return new FrameLibDetectSchema(in, sampleFraction, k).apply();
	}

	private FrameBlock apply() {
		final int cols = in.getNumColumns();
		final FrameBlock fb = new FrameBlock(UtilFunctions.nCopies(cols, ValueType.STRING));
		String[] schemaInfo = (k == 1) ? singleThreadApply() : parallelApply();
		fb.appendRow(schemaInfo);
		return fb;
	}

	private String[] singleThreadApply() {
		final int cols = in.getNumColumns();
		final String[] schemaInfo = new String[cols];

		for(int i = 0; i < cols; i++)
			assign(schemaInfo, in.getColumn(i).analyzeValueType(sampleSize), i);

		return schemaInfo;
	}

	private String[] parallelApply() {
		final ExecutorService pool = CommonThreadPool.get(k);
		try {
			final int cols = in.getNumColumns();
			final ArrayList<DetectValueTypeTask> tasks = new ArrayList<>(cols);
			for(int i = 0; i < cols; i++)
				tasks.add(new DetectValueTypeTask(in.getColumn(i), sampleSize));
			final List<Future<Pair<ValueType, Boolean>>> ret = pool.invokeAll(tasks);
			final String[] schemaInfo = new String[cols];
			pool.shutdown();
			for(int i = 0; i < cols; i++)
				assign(schemaInfo, ret.get(i).get(), i);

			return schemaInfo;
		}
		catch(ExecutionException | InterruptedException e) {
			pool.shutdown();
			throw new DMLRuntimeException("Exception interrupted or exception thrown in detectSchema", e);
		}
	}

	private static void assign(String[] schemaInfo, Pair<ValueType, Boolean> v, int i) {

		if(v.getValue())
			schemaInfo[i] = v.getKey() + FrameUtil.SCHEMA_SEPARATOR + "n";
		else
			schemaInfo[i] = v.getKey().toString();
	}

	private static class DetectValueTypeTask implements Callable<Pair<ValueType, Boolean>> {
		private final Array<?> _obj;
		final int _nCells;

		protected DetectValueTypeTask(Array<?> obj, int nCells) {
			_obj = obj;
			_nCells = nCells;
		}

		@Override
		public Pair<ValueType, Boolean> call() {
			return _obj.analyzeValueType(_nCells);
		}
	}

}
