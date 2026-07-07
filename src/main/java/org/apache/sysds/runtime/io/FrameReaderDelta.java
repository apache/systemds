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

package org.apache.sysds.runtime.io;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;

import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.columns.Array;
import org.apache.sysds.runtime.frame.data.columns.ArrayFactory;

import io.delta.kernel.data.ColumnVector;
import io.delta.kernel.data.Row;
import io.delta.kernel.engine.Engine;
import io.delta.kernel.types.DataType;

/**
 * Single-threaded native Delta Lake reader for frames, built on the Spark-free Delta Kernel library. It opens the
 * latest snapshot of a Delta table through the kernel (log replay, schema, data-file listing) and decodes the parquet
 * data files directly into pre-allocated columns via parquet-mr's column API; tables with deletion vectors, partition
 * columns or missing row statistics are read through the kernel's default engine instead (which applies deletion
 * vectors and splices partition values). Schema and column names are derived from the Delta table schema.
 *
 * <p>
 * Data is extracted column-at-a-time into primitive arrays (no per-cell boxing or {@code FrameBlock.set} dispatch) and
 * the frame is constructed directly from typed column {@link Array}s. Supported column types map to SystemDS value
 * types: double, float, long, int, short, byte, boolean, and string. Neither the schema nor the dimensions need to be
 * supplied; they are discovered from the table.
 * </p>
 */
public class FrameReaderDelta extends FrameReader {

	// per-column read codes (how to pull a value out of the Delta column vector);
	// aliases of the shared codes in DeltaKernelUtils so the frame read dispatch stays
	// in lockstep with the matrix reader's type mapping. Package visible so the parallel
	// reader can reuse the same dispatch.
	static final int R_DOUBLE = DeltaKernelUtils.T_DOUBLE, R_FLOAT = DeltaKernelUtils.T_FLOAT,
		R_LONG = DeltaKernelUtils.T_LONG, R_INT = DeltaKernelUtils.T_INT, R_SHORT = DeltaKernelUtils.T_SHORT,
		R_BYTE = DeltaKernelUtils.T_BYTE, R_BOOLEAN = DeltaKernelUtils.T_BOOLEAN, R_STRING = DeltaKernelUtils.T_STRING;

	@Override
	public FrameBlock readFrameFromHDFS(String fname, ValueType[] schema, String[] names, long rlen, long clen)
		throws IOException, DMLRuntimeException {
		Engine engine = DeltaKernelUtils.createEngine();
		String tablePath = DeltaKernelUtils.qualify(fname);
		DeltaKernelUtils.ScanHandle handle = DeltaKernelUtils.openScan(engine, tablePath);
		return readWithHandle(fname, engine, handle);
	}

	/**
	 * Materialize the frame from an already-opened engine and scan handle. Factored out so the parallel reader can
	 * reuse a handle it already opened for its single-file/single-thread fallback instead of re-opening the (expensive)
	 * Delta snapshot a second time.
	 *
	 * @param fname  the table path (for error messages)
	 * @param engine the Delta Kernel engine
	 * @param handle the opened scan handle
	 * @return the materialized frame block
	 */
	protected FrameBlock readWithHandle(String fname, Engine engine, DeltaKernelUtils.ScanHandle handle)
		throws IOException {
		final ReadPlan plan = planColumns(handle);

		// fast path: exact per-file row counts are known from metadata (no deletion
		// vectors) -> pre-size one typed array per column and decode each file
		// straight into its row offset, avoiding the per-batch extract + concatenate.
		if(useDirectPath(handle)) {
			long total = 0;
			for(long r : handle.numRecords)
				total += r;
			// empty table: the typed column arrays cannot be zero-length, so return a
			// schema-only frame with the discovered schema/names and zero rows.
			if(total == 0)
				return new FrameBlock(plan.vt, plan.cnames, 0);
			if(total <= Integer.MAX_VALUE)
				return readDirect(fname, handle, plan, (int) total);
		}

		// fallback: row counts unknown or deletion vectors present -> decode into
		// per-batch arrays and concatenate per column in file order.
		return readBuffered(engine, handle, plan);
	}

	/**
	 * Immutable per-column read plan derived once from the Delta table schema: how to pull each column out of the
	 * kernel column vector ({@code readCodes}), the resulting SystemDS value types, and the column names. Shared by the
	 * serial and parallel readers so the schema-to-column mapping lives in exactly one place.
	 */
	protected static final class ReadPlan {
		final int ncol;
		final int[] readCodes;
		final ValueType[] vt;
		final String[] cnames;

		private ReadPlan(int ncol, int[] readCodes, ValueType[] vt, String[] cnames) {
			this.ncol = ncol;
			this.readCodes = readCodes;
			this.vt = vt;
			this.cnames = cnames;
		}
	}

	/** Derive the {@link ReadPlan} (read codes, value types, names) from the opened scan handle's schema. */
	protected static ReadPlan planColumns(DeltaKernelUtils.ScanHandle handle) {
		final int ncol = handle.schema.length();
		final int[] readCodes = new int[ncol];
		final ValueType[] vt = new ValueType[ncol];
		final String[] cnames = new String[ncol];
		for(int c = 0; c < ncol; c++) {
			DataType dt = handle.schema.at(c).getDataType();
			readCodes[c] = readCode(dt, handle.schema.at(c).getName());
			vt[c] = valueType(readCodes[c]);
			cnames[c] = handle.schema.at(c).getName();
		}
		return new ReadPlan(ncol, readCodes, vt, cnames);
	}

	/**
	 * Whether the metadata-driven direct read fast path can be used for this table: exact per-file row counts (no
	 * deletion vectors), so the output can be pre-sized, and a physical read schema that maps 1:1 onto the output
	 * columns (no partition columns or kernel metadata columns to splice back in), so each data file can be decoded
	 * straight into its row offset without the kernel engine. The buffered kernel-path fallback covers everything else
	 * (deletion vectors, missing statistics, partitioned tables).
	 *
	 * @param handle the opened scan handle
	 * @return true if the direct path is applicable
	 */
	protected boolean useDirectPath(DeltaKernelUtils.ScanHandle handle) {
		return handle.hasExactRowCounts() &&
			DeltaKernelUtils.supportsDirectDecode(handle.schema, handle.physicalReadSchema);
	}

	/**
	 * Fast path: decode each data file straight into pre-sized typed column arrays at a metadata-derived row offset,
	 * through parquet-mr's column API with no kernel engine or intermediate batch vectors in the path. One allocation
	 * per column, single pass, no per-batch buffers or serial concatenation.
	 */
	private FrameBlock readDirect(String fname, DeltaKernelUtils.ScanHandle handle, ReadPlan plan, int nrow)
		throws IOException {
		final int ncol = plan.ncol;
		final int[] readCodes = plan.readCodes;
		final Object[] dest = new Object[ncol];
		for(int c = 0; c < ncol; c++)
			dest[c] = ArrayFactory.allocateBacking(plan.vt[c], nrow);

		int base = 0;
		for(int i = 0; i < handle.scanFiles.size(); i++) {
			// exclusive upper row bound for this file's slice (enforced inside the
			// decode before each row group is written)
			final int limit = base + (int) handle.numRecords[i];
			String path = DeltaKernelUtils.dataFilePath(handle.scanFiles.get(i));
			int n = DeltaKernelUtils.decodeDataFileInto(path, handle.physicalReadSchema, readCodes, dest, base, limit,
				fname);
			// fail loud on underflow: a file decoding fewer rows than its numRecords
			// statistic would leave the tail of the slice at the array default (0/null)
			// while nrow still reports the (inflated) statistic.
			if(base + n != limit)
				throw new DMLRuntimeException("Delta file produced " + n + " rows, expected " + (limit - base)
					+ " from its numRecords statistic; refusing direct read of " + fname);
			base = limit;
		}

		Array<?>[] columns = new Array<?>[ncol];
		for(int c = 0; c < ncol; c++)
			columns[c] = ArrayFactory.create(plan.vt[c], dest[c]);
		FrameBlock ret = new FrameBlock(columns);
		ret.setColumnNames(plan.cnames);
		return ret;
	}

	/**
	 * Fallback path: decode each batch into per-batch typed arrays and concatenate them per column in file order. Used
	 * when exact per-file row counts are not available (missing statistics or deletion vectors present), so the output
	 * cannot be pre-sized up front.
	 */
	private FrameBlock readBuffered(Engine engine, DeltaKernelUtils.ScanHandle handle, ReadPlan plan)
		throws IOException {
		final int ncol = plan.ncol;
		final int[] readCodes = plan.readCodes;
		final ArrayList<Object[]> batchCols = new ArrayList<>();
		final ArrayList<Integer> batchSizes = new ArrayList<>();
		final int[] nrowHolder = new int[1];
		for(Row scanFileRow : handle.scanFiles) {
			DeltaKernelUtils.readScanFile(engine, handle.scanState, handle.physicalReadSchema, scanFileRow,
				(cols, size, selected) -> {
					int n = DeltaKernelUtils.countSelected(size, selected);
					Object[] extracted = new Object[ncol];
					for(int c = 0; c < ncol; c++) {
						// decode into a fresh per-batch array via the shared alloc +
						// decode primitives (the same ones the direct path uses)
						Object col = ArrayFactory.allocateBacking(plan.vt[c], n);
						extractColumnInto(cols[c], size, selected, readCodes[c], col, 0);
						extracted[c] = col;
					}
					batchCols.add(extracted);
					batchSizes.add(n);
					nrowHolder[0] += n;
				});
		}

		int nrow = nrowHolder[0];
		// empty table: return a schema-only frame with the discovered schema/names.
		if(nrow == 0)
			return new FrameBlock(plan.vt, plan.cnames, 0);
		Array<?>[] columns = new Array<?>[ncol];
		for(int c = 0; c < ncol; c++)
			columns[c] = concatColumn(plan.vt[c], nrow, batchCols, batchSizes, c);
		FrameBlock ret = new FrameBlock(columns);
		ret.setColumnNames(plan.cnames);
		return ret;
	}

	/**
	 * Concatenate the per-batch typed arrays of one column (in file/batch order) into a single pre-sized array and wrap
	 * it as a frame {@link Array}. The copy is type-agnostic ({@link System#arraycopy} works on the boxed primitive or
	 * object arrays), so there is no per-type dispatch here: allocation and wrapping reuse
	 * {@link ArrayFactory#allocateBacking(ValueType, int)} and {@link ArrayFactory#create(ValueType, Object)}, the same
	 * primitives the single-pass direct path uses.
	 *
	 * <p>
	 * Only the buffered fallback needs this concatenation; the default direct path decodes straight into one pre-sized
	 * array per column with no intermediate per-batch arrays.
	 * </p>
	 */
	static Array<?> concatColumn(ValueType vt, int nrow, ArrayList<Object[]> batchCols, ArrayList<Integer> batchSizes,
		int c) {
		Object full = ArrayFactory.allocateBacking(vt, nrow);
		int off = 0;
		for(int b = 0; b < batchCols.size(); b++) {
			int n = batchSizes.get(b);
			System.arraycopy(batchCols.get(b)[c], 0, full, off, n);
			off += n;
		}
		return ArrayFactory.create(vt, full);
	}

	static int readCode(DataType dt, String name) {
		// reuse the shared Delta type -> code mapping; frames additionally reject the
		// types the matrix reader also cannot map (typeCode returns -1)
		int code = DeltaKernelUtils.typeCode(dt);
		if(code < 0)
			throw new DMLRuntimeException(
				"Unsupported non-mappable Delta column '" + name + "' of type " + dt + " for frame read.");
		return code;
	}

	static ValueType valueType(int readCode) {
		switch(readCode) {
			case R_DOUBLE:
				return ValueType.FP64;
			case R_FLOAT:
				return ValueType.FP32;
			case R_LONG:
				return ValueType.INT64;
			case R_INT:
			case R_SHORT:
			case R_BYTE:
				return ValueType.INT32;
			case R_BOOLEAN:
				return ValueType.BOOLEAN;
			default:
				return ValueType.STRING;
		}
	}

	/**
	 * Decode the live (selected, after deletion vector) rows of one column batch directly into a pre-sized typed array
	 * starting at absolute row {@code destOff}. Null numeric cells keep the array default (0); string nulls are stored
	 * as null.
	 */
	static void extractColumnInto(ColumnVector col, int size, boolean[] selected, int readCode, Object dest,
		int destOff) {
		switch(readCode) {
			case R_DOUBLE: {
				double[] a = (double[]) dest;
				int lr = destOff;
				for(int r = 0; r < size; r++) {
					if(selected != null && !selected[r])
						continue;
					if(!col.isNullAt(r))
						a[lr] = col.getDouble(r);
					lr++;
				}
				break;
			}
			case R_FLOAT: {
				float[] a = (float[]) dest;
				int lr = destOff;
				for(int r = 0; r < size; r++) {
					if(selected != null && !selected[r])
						continue;
					if(!col.isNullAt(r))
						a[lr] = col.getFloat(r);
					lr++;
				}
				break;
			}
			case R_LONG: {
				long[] a = (long[]) dest;
				int lr = destOff;
				for(int r = 0; r < size; r++) {
					if(selected != null && !selected[r])
						continue;
					if(!col.isNullAt(r))
						a[lr] = col.getLong(r);
					lr++;
				}
				break;
			}
			case R_INT: {
				int[] a = (int[]) dest;
				int lr = destOff;
				for(int r = 0; r < size; r++) {
					if(selected != null && !selected[r])
						continue;
					if(!col.isNullAt(r))
						a[lr] = col.getInt(r);
					lr++;
				}
				break;
			}
			case R_SHORT: {
				int[] a = (int[]) dest;
				int lr = destOff;
				for(int r = 0; r < size; r++) {
					if(selected != null && !selected[r])
						continue;
					if(!col.isNullAt(r))
						a[lr] = col.getShort(r);
					lr++;
				}
				break;
			}
			case R_BYTE: {
				int[] a = (int[]) dest;
				int lr = destOff;
				for(int r = 0; r < size; r++) {
					if(selected != null && !selected[r])
						continue;
					if(!col.isNullAt(r))
						a[lr] = col.getByte(r);
					lr++;
				}
				break;
			}
			case R_BOOLEAN: {
				boolean[] a = (boolean[]) dest;
				int lr = destOff;
				for(int r = 0; r < size; r++) {
					if(selected != null && !selected[r])
						continue;
					if(!col.isNullAt(r))
						a[lr] = col.getBoolean(r);
					lr++;
				}
				break;
			}
			default: { // R_STRING
				String[] a = (String[]) dest;
				int lr = destOff;
				for(int r = 0; r < size; r++) {
					if(selected != null && !selected[r])
						continue;
					a[lr] = col.isNullAt(r) ? null : col.getString(r);
					lr++;
				}
				break;
			}
		}
	}

	@Override
	public FrameBlock readFrameFromInputStream(InputStream is, ValueType[] schema, String[] names, long rlen, long clen)
		throws IOException, DMLRuntimeException {
		throw new UnsupportedOperationException(
			"Reading a Delta table from an input stream is not supported; Delta is a directory-based table format.");
	}
}
