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
import java.util.HashMap;
import java.util.List;
import java.util.Map.Entry;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.columns.ACompressedArray;
import org.apache.sysds.runtime.frame.data.columns.Array;
import org.apache.sysds.runtime.frame.data.columns.ArrayFactory;
import org.apache.sysds.runtime.frame.data.columns.ColumnMetadata;
import org.apache.sysds.runtime.matrix.data.Pair;
import org.apache.sysds.runtime.util.CommonThreadPool;

public class FrameLibApplySchema {

	protected static final Log LOG = LogFactory.getLog(FrameLibApplySchema.class.getName());

	public static int PAR_ROW_THRESHOLD = 1024;

	private final FrameBlock fb;
	private final ValueType[] schema;
	private final boolean[] nulls;
	private final int nCol;
	private final int nRow;
	private final Array<?>[] columnsIn;
	private final Array<?>[] columnsOut;

	private final int k;

	public static FrameBlock applySchema(FrameBlock fb, FrameBlock schema) {
		return applySchema(fb, schema, 1);
	}

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
		nRow = fb.getNumRows();
		columnsIn = fb.getColumns();
		columnsOut = new Array<?>[nCol];
	}

	private FrameBlock apply() {

		if(k <= 1 || nCol == 1 || containsCompressed())
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

		FrameBlock out = new FrameBlock(schema, colNames, meta, columnsOut);
		if(LOG.isDebugEnabled()) {

			long inMem = fb.getInMemorySize();
			long outMem = out.getInMemorySize();
			LOG.debug(String.format("Schema Apply Input Size: %16d", inMem));
			LOG.debug(String.format("            Output Size: %16d", outMem));
			LOG.debug(String.format("            Ratio      : %4.3f", ((double) inMem / outMem)));
		}
		return out;
	}

	private boolean containsCompressed(){
		for(Array<?> col : fb.getColumns())
			if(col instanceof ACompressedArray)
				return true;
		
		return false;
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

	/**
	 * Try to change the value type in the range given.
	 * 
	 * @param j The column index
	 * @param l The lower bound
	 * @param u The upper bound
	 * @return j if the call failed otherwise -1.
	 */
	private int tryChangeType(int j, int l, int u) {
		try {
			columnsIn[j].changeTypeWithNulls(columnsOut[j], l, u);
			return -1;
		}
		catch(Exception e) {
			LOG.warn(e.getMessage());
			return j;
		}
	}

	private void applyMultiThread() {
		final ExecutorService pool = CommonThreadPool.get(k);
		try {
			List<Future<Integer>> f = new ArrayList<>(nCol);

			final int rowThreads = Math.max(1, (k * 2) / nCol);
			final int block = Math.max(((nRow / rowThreads) / 64) * 64, PAR_ROW_THRESHOLD);
			for(int i = 0; i < nCol; i++) {
				final int j = i; // final col variable for task
				if(schema[j] == columnsIn[i].getValueType() || block > nRow) {
					apply(j);
				}
				else {

					if(nulls != null && nulls[j])
						columnsOut[j] = ArrayFactory.allocateOptional(schema[j], nRow);
					else
						columnsOut[j] = ArrayFactory.allocate(schema[j], nRow);
					for(int r = 0; r < nRow; r += block) {
						final int start = r;
						final int end = Math.min(nRow, r + block);
						f.add(pool.submit(() -> tryChangeType(j, start, end)));
					}
				}
			}

			final HashMap<Integer, Array<?>> fixes = new HashMap<>();

			for(Future<Integer> e : f) {
				final int j = e.get();
				if(j >= 0 && !fixes.containsKey(j)) {
					Pair<ValueType, Boolean> sc = columnsIn[j].analyzeValueType();
					LOG.warn("Failed to change type of column: " + j + " sample said value type: " + schema[j]
						+ " Full analysis says: " + sc.getKey());
					final Array<?> tmp;
					if(sc.getValue())
						tmp = ArrayFactory.allocateOptional(sc.getKey(), nRow);
					else
						tmp = ArrayFactory.allocate(sc.getKey(), nRow);
					columnsIn[j].changeType(tmp);
					fixes.put(j, tmp);
				}
			}

			for(Entry<Integer, Array<?>> e : fixes.entrySet()) {
				columnsOut[e.getKey()] = e.getValue();
			}
		}
		catch(InterruptedException | ExecutionException e) {
			throw new DMLRuntimeException("Failed to combine column groups", e);
		}
		finally {
			pool.shutdown();

		}
	}

	private void verifySize() {
		if(schema.length != fb.getSchema().length)
			throw new DMLRuntimeException(//
				"Invalid apply schema with different number of columns expected: " + fb.getSchema().length + " got: "
					+ schema.length);
	}
}
