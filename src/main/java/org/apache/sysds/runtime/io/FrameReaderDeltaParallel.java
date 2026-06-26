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
import java.util.ArrayList;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.columns.Array;
import org.apache.sysds.runtime.frame.data.columns.ArrayFactory;
import org.apache.sysds.runtime.util.CommonThreadPool;

import io.delta.kernel.data.ColumnVector;
import io.delta.kernel.data.Row;
import io.delta.kernel.engine.Engine;
import io.delta.kernel.types.DataType;

/**
 * Parallel native Delta Lake frame reader. Delta tables are stored as one or
 * more parquet data files; this reader decodes those files concurrently (one
 * task per data file) and assembles them into a column-major {@link FrameBlock}
 * in the original file order.
 *
 * <p>It mirrors {@link ReaderDeltaParallel} (the matrix variant) but produces
 * typed column {@link Array}s instead of a dense {@code double[]}. As with the
 * matrix reader, the expensive part of a Delta read is the per-file parquet
 * decode, so parallelizing across data files is the natural speedup. A table
 * backed by a single data file cannot be split this way, so the reader
 * transparently falls back to the sequential {@link FrameReaderDelta}.</p>
 */
public class FrameReaderDeltaParallel extends FrameReaderDelta {

	private final int _numThreads;

	public FrameReaderDeltaParallel() {
		_numThreads = OptimizerUtils.getParallelBinaryReadParallelism();
	}

	@Override
	public FrameBlock readFrameFromHDFS(String fname, ValueType[] schema, String[] names, long rlen, long clen)
		throws IOException, DMLRuntimeException
	{
		Engine engine = DeltaKernelUtils.createEngine();
		String tablePath = DeltaKernelUtils.qualify(fname);
		DeltaKernelUtils.ScanHandle handle = DeltaKernelUtils.openScan(engine, tablePath);

		final int nfiles = handle.scanFiles.size();
		//nothing to gain from parallelism for single-file (or empty) tables
		if( _numThreads <= 1 || nfiles <= 1 )
			return super.readFrameFromHDFS(fname, schema, names, rlen, clen);

		//derive per-column read codes, value types and names once from the schema
		final int ncol = handle.schema.length();
		final int[] readCodes = new int[ncol];
		final ValueType[] vt = new ValueType[ncol];
		final String[] cnames = new String[ncol];
		for( int c=0; c<ncol; c++ ) {
			DataType dt = handle.schema.at(c).getDataType();
			readCodes[c] = readCode(dt, handle.schema.at(c).getName());
			vt[c] = valueType(readCodes[c]);
			cnames[c] = handle.schema.at(c).getName();
		}

		//fast path: exact per-file row counts are known from metadata -> pre-size
		//one typed array per column and let each thread decode directly into its
		//row offset (no intermediate buffers, no serial concatenation).
		if( useDirectPath(handle) ) {
			long total = 0;
			for( long r : handle.numRecords )
				total += r;
			if( total > 0 && total <= Integer.MAX_VALUE )
				return readDirect(fname, handle, ncol, readCodes, vt, cnames, (int) total);
		}

		return readBuffered(fname, handle, ncol, readCodes, vt, cnames);
	}

	/**
	 * Whether the metadata-driven direct-write fast path can be used for this
	 * table (exact per-file row counts and no deletion vectors). Visible for
	 * testing: the buffered fallback is otherwise only reachable for tables
	 * lacking row statistics or carrying deletion vectors, which the SystemDS
	 * Delta writer never produces.
	 *
	 * @param handle the opened scan handle
	 * @return true if the direct path is applicable
	 */
	protected boolean useDirectPath(DeltaKernelUtils.ScanHandle handle) {
		return handle.hasExactRowCounts();
	}

	/**
	 * Fast path: each thread decodes one data file straight into the final typed
	 * column arrays at a metadata-derived row offset. Single allocation per
	 * column, fully parallel.
	 */
	private FrameBlock readDirect(String fname, DeltaKernelUtils.ScanHandle handle,
		int ncol, int[] readCodes, ValueType[] vt, String[] cnames, int nrow) throws IOException
	{
		final int nfiles = handle.scanFiles.size();
		final int[] rowOffset = new int[nfiles];
		int acc = 0;
		for( int i=0; i<nfiles; i++ ) {
			rowOffset[i] = acc;
			acc += (int) handle.numRecords[i];
		}

		//pre-size one typed array per column for the whole table
		final Object[] dest = new Object[ncol];
		for( int c=0; c<ncol; c++ )
			dest[c] = allocColumn(vt[c], nrow);

		//request full parallelism (returns the shared common pool); the number of
		//tasks (one per data file) naturally caps concurrency. Avoids the per-thread
		//pool-size caching in CommonThreadPool.get(k) that could otherwise throttle
		//this reader to a smaller pool created earlier on the same thread.
		final ExecutorService pool = CommonThreadPool.get(_numThreads);
		try {
			ArrayList<Callable<Object>> tasks = new ArrayList<>(nfiles);
			for( int i=0; i<nfiles; i++ ) {
				final Row scanFileRow = handle.scanFiles.get(i);
				final int base = rowOffset[i];
				tasks.add(() -> {
					int[] cur = new int[] {base};
					Engine eng = DeltaKernelUtils.createEngine();
					DeltaKernelUtils.readScanFile(eng, handle.scanState, handle.physicalReadSchema, scanFileRow,
						(cols, size, selected) -> {
							for( int c=0; c<ncol; c++ )
								extractColumnInto(cols[c], size, selected, readCodes[c], dest[c], cur[0]);
							cur[0] += DeltaKernelUtils.countSelected(size, selected);
						});
					return null;
				});
			}
			for( Future<Object> f : pool.invokeAll(tasks) )
				f.get();
		}
		catch(Exception ex) {
			throw new IOException("Failed parallel read of Delta table: " + fname, ex);
		}
		finally {
			pool.shutdown();
		}

		Array<?>[] columns = new Array<?>[ncol];
		for( int c=0; c<ncol; c++ )
			columns[c] = createColumn(vt[c], dest[c]);

		FrameBlock ret = new FrameBlock(columns);
		ret.setColumnNames(cnames);
		return ret;
	}

	/**
	 * Fallback path: decode each file in parallel into per-file per-column batch
	 * arrays (used when row counts are unknown or deletion vectors are present),
	 * then concatenate per column in file order via the shared
	 * {@link FrameReaderDelta#buildColumn} helper.
	 */
	private FrameBlock readBuffered(String fname, DeltaKernelUtils.ScanHandle handle,
		int ncol, int[] readCodes, ValueType[] vt, String[] cnames) throws IOException
	{
		final int nfiles = handle.scanFiles.size();
		@SuppressWarnings("unchecked")
		final ArrayList<Object[]>[] fileCols = new ArrayList[nfiles];
		@SuppressWarnings("unchecked")
		final ArrayList<Integer>[] fileSizes = new ArrayList[nfiles];
		final ExecutorService pool = CommonThreadPool.get(_numThreads);
		try {
			ArrayList<Callable<Object>> tasks = new ArrayList<>(nfiles);
			for( int i=0; i<nfiles; i++ ) {
				final int fi = i;
				final Row scanFileRow = handle.scanFiles.get(i);
				tasks.add(() -> {
					ArrayList<Object[]> bcs = new ArrayList<>();
					ArrayList<Integer> bss = new ArrayList<>();
					Engine eng = DeltaKernelUtils.createEngine();
					DeltaKernelUtils.readScanFile(eng, handle.scanState, handle.physicalReadSchema, scanFileRow,
						(cols, size, selected) -> {
							int n = DeltaKernelUtils.countSelected(size, selected);
							Object[] extracted = new Object[ncol];
							for( int c=0; c<ncol; c++ )
								extracted[c] = extractColumn(cols[c], size, selected, n, readCodes[c]);
							bcs.add(extracted);
							bss.add(n);
						});
					fileCols[fi] = bcs;
					fileSizes[fi] = bss;
					return null;
				});
			}
			for( Future<Object> f : pool.invokeAll(tasks) )
				f.get();
		}
		catch(Exception ex) {
			throw new IOException("Failed parallel read of Delta table: " + fname, ex);
		}
		finally {
			pool.shutdown();
		}

		//flatten the per-file batches in file order and concatenate per column
		ArrayList<Object[]> batchCols = new ArrayList<>();
		ArrayList<Integer> batchSizes = new ArrayList<>();
		int nrow = 0;
		for( int i=0; i<nfiles; i++ ) {
			batchCols.addAll(fileCols[i]);
			batchSizes.addAll(fileSizes[i]);
			for( int n : fileSizes[i] )
				nrow += n;
		}

		Array<?>[] columns = new Array<?>[ncol];
		for( int c=0; c<ncol; c++ )
			columns[c] = buildColumn(vt[c], nrow, batchCols, batchSizes, c);

		FrameBlock ret = new FrameBlock(columns);
		ret.setColumnNames(cnames);
		return ret;
	}

	/** Allocate a pre-sized typed column array matching the target value type. */
	private static Object allocColumn(ValueType vt, int n) {
		switch( vt ) {
			case FP64:    return new double[n];
			case FP32:    return new float[n];
			case INT64:   return new long[n];
			case INT32:   return new int[n];
			case BOOLEAN: return new boolean[n];
			default:      return new String[n]; // STRING
		}
	}

	/** Wrap a fully populated typed column array into a frame {@link Array}. */
	private static Array<?> createColumn(ValueType vt, Object full) {
		switch( vt ) {
			case FP64:    return ArrayFactory.create((double[]) full);
			case FP32:    return ArrayFactory.create((float[]) full);
			case INT64:   return ArrayFactory.create((long[]) full);
			case INT32:   return ArrayFactory.create((int[]) full);
			case BOOLEAN: return ArrayFactory.create((boolean[]) full);
			default:      return ArrayFactory.create((String[]) full); // STRING
		}
	}

	/**
	 * Decode the live (selected, after deletion vector) rows of one column batch
	 * directly into a pre-sized typed array starting at absolute row {@code destOff}.
	 * Null numeric cells keep the array default (0); string nulls are stored as null.
	 */
	private static void extractColumnInto(ColumnVector col, int size, boolean[] selected,
		int readCode, Object dest, int destOff)
	{
		switch( readCode ) {
			case R_DOUBLE: {
				double[] a = (double[]) dest;
				int lr = destOff;
				for( int r=0; r<size; r++ ) {
					if( selected != null && !selected[r] ) continue;
					if( !col.isNullAt(r) ) a[lr] = col.getDouble(r);
					lr++;
				}
				break;
			}
			case R_FLOAT: {
				float[] a = (float[]) dest;
				int lr = destOff;
				for( int r=0; r<size; r++ ) {
					if( selected != null && !selected[r] ) continue;
					if( !col.isNullAt(r) ) a[lr] = col.getFloat(r);
					lr++;
				}
				break;
			}
			case R_LONG: {
				long[] a = (long[]) dest;
				int lr = destOff;
				for( int r=0; r<size; r++ ) {
					if( selected != null && !selected[r] ) continue;
					if( !col.isNullAt(r) ) a[lr] = col.getLong(r);
					lr++;
				}
				break;
			}
			case R_INT: {
				int[] a = (int[]) dest;
				int lr = destOff;
				for( int r=0; r<size; r++ ) {
					if( selected != null && !selected[r] ) continue;
					if( !col.isNullAt(r) ) a[lr] = col.getInt(r);
					lr++;
				}
				break;
			}
			case R_SHORT: {
				int[] a = (int[]) dest;
				int lr = destOff;
				for( int r=0; r<size; r++ ) {
					if( selected != null && !selected[r] ) continue;
					if( !col.isNullAt(r) ) a[lr] = col.getShort(r);
					lr++;
				}
				break;
			}
			case R_BYTE: {
				int[] a = (int[]) dest;
				int lr = destOff;
				for( int r=0; r<size; r++ ) {
					if( selected != null && !selected[r] ) continue;
					if( !col.isNullAt(r) ) a[lr] = col.getByte(r);
					lr++;
				}
				break;
			}
			case R_BOOLEAN: {
				boolean[] a = (boolean[]) dest;
				int lr = destOff;
				for( int r=0; r<size; r++ ) {
					if( selected != null && !selected[r] ) continue;
					if( !col.isNullAt(r) ) a[lr] = col.getBoolean(r);
					lr++;
				}
				break;
			}
			default: { // R_STRING
				String[] a = (String[]) dest;
				int lr = destOff;
				for( int r=0; r<size; r++ ) {
					if( selected != null && !selected[r] ) continue;
					a[lr] = col.isNullAt(r) ? null : col.getString(r);
					lr++;
				}
				break;
			}
		}
	}
}
