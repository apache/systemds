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
	private final boolean[] nulls;
	private final int nCol;
	private final Array<?>[] columnsIn;
	private final Array<?>[] columnsOut;

	private final int k;

	/**
	 * Method to create a new FrameBlock where the given schema is applied, k is the parallelization degree.
	 * 
	 * @param fb     The input block to apply schema to
	 * @param schema The schema to apply
	 * @param k      The parallelization degree
	 * @return A new FrameBlock allocated with new arrays.
	 */
	public static FrameBlock applySchema(FrameBlock fb, FrameBlock schema, int k) {
		// apply frame schema from DML
		ValueType[] sv = new ValueType[schema.getNumColumns()];
		boolean[] nulls = new boolean[schema.getNumColumns()];
		for(int i = 0; i < schema.getNumColumns(); i++) {
			final String[] v = schema.get(0, i).toString().split(FrameUtil.SCHEMA_SEPARATOR);
			nulls[i] = v.length == 2 && v[1].equals("n");
			sv[i] = ValueType.fromExternalString(v[0]);
		}

		return new FrameLibApplySchema(fb, sv, nulls, k).apply();
	}

	/**
	 * Method to create a new FrameBlock where the given schema is applied.
	 * 
	 * @param fb     The input block to apply schema to
	 * @param schema The schema to apply
	 * @return A new FrameBlock allocated with new arrays.
	 */
	public static FrameBlock applySchema(FrameBlock fb, ValueType[] schema) {
		return new FrameLibApplySchema(fb, schema, null, 1).apply();
	}

	/**
	 * Method to create a new FrameBlock where the given schema is applied, k is the parallelization degree.
	 * 
	 * @param fb     The input block to apply schema to
	 * @param schema The schema to apply
	 * @param k      The parallelization degree
	 * @return A new FrameBlock allocated with new arrays.
	 */
	public static FrameBlock applySchema(FrameBlock fb, ValueType[] schema, int k) {
		return new FrameLibApplySchema(fb, schema, null, k).apply();
	}

	private FrameLibApplySchema(FrameBlock fb, ValueType[] schema, boolean[] nulls, int k) {
		this.fb = fb;
		this.schema = schema;
		this.nulls = nulls;
		this.k = k;
		verifySize();
		nCol = fb.getNumColumns();
		columnsIn = fb.getColumns();
		columnsOut = new Array<?>[nCol];
	}

	private FrameBlock apply() {
		if(k <= 1 || nCol == 1)
			applySingleThread();
		else
			applyMultiThread();

		boolean same = true;
		for(int i = 0; i < columnsIn.length && same; i++)
			same = columnsIn[i] == columnsOut[i];

		if(same)
			return this.fb;

		final String[] colNames = fb.getColumnNames(false);
		final ColumnMetadata[] meta = fb.getColumnMetadata();
		return new FrameBlock(schema, colNames, meta, columnsOut);
	}

	private void applySingleThread() {
		for(int i = 0; i < nCol; i++)
			apply(i);
	}

	private void apply(int i) {
		if(nulls != null)
			columnsOut[i] = nulls[i] ? //
				columnsIn[i].changeTypeWithNulls(schema[i]) : //
				columnsIn[i].changeType(schema[i]);
		else
			columnsOut[i] = columnsIn[i].changeType(schema[i]);

	}

	private void applyMultiThread() {
		final ExecutorService pool = CommonThreadPool.get(k);
		try {

			pool.submit(() -> {
				IntStream.rangeClosed(0, nCol - 1).parallel() // parallel columns
					.forEach(x -> apply(x));
			}).get();

			pool.shutdown();
		}
		catch(InterruptedException | ExecutionException e) {
			pool.shutdown();
			throw new DMLRuntimeException("Failed to combine column groups", e);
		}
	}

	private void verifySize() {
		if(schema.length != fb.getSchema().length)
			throw new DMLRuntimeException(//
				"Invalid apply schema with different number of columns expected: " + fb.getSchema().length + " got: "
					+ schema.length);
	}
}
