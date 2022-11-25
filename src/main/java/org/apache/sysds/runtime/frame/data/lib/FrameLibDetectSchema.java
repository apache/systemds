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

import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.columns.Array;
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.apache.sysds.runtime.util.UtilFunctions;

public final class FrameLibDetectSchema {
	// private static final Log LOG = LogFactory.getLog(FrameBlock.class.getName());

	private FrameLibDetectSchema() {
		// private constructor
	}

	public static FrameBlock detectSchema(FrameBlock in) {
		return detectSchema(in, 1.0);
	}

	public static FrameBlock detectSchema(FrameBlock in, double sampleFraction) {
		// LOG.error(Arrays.toString(in.getSchema()));
		final int cols = in.getNumColumns();
		ArrayList<DetectValueTypeTask> tasks = new ArrayList<>(cols);
		for(int i = 0; i < cols; i++)
			tasks.add(new DetectValueTypeTask(in.getColumn(i)));

		List<Future<ValueType>> ret;

		ExecutorService pool = CommonThreadPool.get(cols);
		try {
			ret = pool.invokeAll(tasks);
			final FrameBlock fb = new FrameBlock(UtilFunctions.nCopies(cols, ValueType.STRING));
			final String[] schemaInfo = new String[cols];
			pool.shutdown();
			for(int i = 0; i < cols; i++)
				schemaInfo[i] = ret.get(i).get().toString();

			fb.appendRow(schemaInfo);
			return fb;
		}
		catch(ExecutionException | InterruptedException e) {
			throw new DMLRuntimeException("Exception Interupted or Exception thrown in Detect Schema", e);
		}
		finally {
			pool.shutdown();
		}
	}

	private static class DetectValueTypeTask implements Callable<ValueType> {
		private final Array<?> _obj;

		protected DetectValueTypeTask(Array<?> obj) {
			_obj = obj;
		}

		@Override
		public ValueType call() {
			return  _obj.analyzeValueType();
		}
	}

}
