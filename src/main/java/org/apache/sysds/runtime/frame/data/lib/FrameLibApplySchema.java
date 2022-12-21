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

import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.stream.IntStream;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.columns.Array;
import org.apache.sysds.runtime.frame.data.columns.ColumnMetadata;
import org.apache.sysds.runtime.util.CommonThreadPool;

public class FrameLibApplySchema {

	protected static final Log LOG = LogFactory.getLog(FrameLibApplySchema.class.getName());

	private final FrameBlock fb;
	private final ValueType[] schema;

	private final int nCol;
	private final Array<?>[] columnsIn;
	private final Array<?>[] columnsOut;

	private FrameLibApplySchema(FrameBlock fb, ValueType[] schema) {
		this.fb = fb;
		this.schema = schema;
		verifySize();
		nCol = fb.getNumColumns();
		columnsIn = fb.getColumns();
		columnsOut = new Array<?>[nCol];

	}

	private FrameBlock apply(int k) {
		if(k <= 1 && nCol > 1)
			applySingleThread();
		else
			applyMultiThread(k);

		final String[] colNames = fb.getColumnNames(false);
		final ColumnMetadata[] meta = fb.getColumnMetadata();
		return new FrameBlock(schema, colNames, meta, columnsOut);
	}

	private void applySingleThread() {
		for(int i = 0; i < nCol; i++)
			columnsOut[i] = columnsIn[i].changeType(schema[i]);
	}

	private void applyMultiThread(int k) {
		final ExecutorService pool = CommonThreadPool.get(k);
		try {

			pool.submit(() -> {
				IntStream.rangeClosed(0, nCol - 1).parallel() // parallel columns
					.forEach(x -> columnsOut[x] = columnsIn[x].changeType(schema[x]));
			}).get();

			pool.shutdown();
		}
		catch(InterruptedException | ExecutionException e) {
			pool.shutdown();
			LOG.error("Failed to combine column groups fallback to single thread", e);
			applySingleThread();
		}
	}

	/**
	 * Method to create a new FrameBlock where the given schema is applied, k is parallelization degree.
	 * 
	 * @param fb     The input block to apply schema to
	 * @param schema The schema to apply
	 * @param k      The parallelization degree
	 * @return A new FrameBlock allocated with new arrays.
	 */
	public static FrameBlock applySchema(FrameBlock fb, ValueType[] schema, int k) {
		return new FrameLibApplySchema(fb, schema).apply(k);
	}

	private void verifySize() {
		if(schema.length != fb.getSchema().length)
			throw new DMLRuntimeException(//
				"Invalid apply schema with different number of columns expected: " + fb.getSchema().length + " got: "
					+ schema.length);
	}
}
