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
import java.util.List;
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

import io.delta.kernel.data.Row;
import io.delta.kernel.engine.Engine;
import io.delta.kernel.types.DataType;

/**
 * Parallel native Delta Lake frame reader. Delta tables are stored as one or more parquet data files; this reader
 * decodes those files concurrently (one task per data file) and assembles them into a column-major {@link FrameBlock}
 * in the original file order.
 *
 * <p>
 * It mirrors {@link ReaderDeltaParallel} (the matrix variant) but produces typed column {@link Array}s instead of a
 * dense {@code double[]}. As with the matrix reader, the expensive part of a Delta read is the per-file parquet decode,
 * so parallelizing across data files is the natural speedup. A table backed by a single data file cannot be split this
 * way, so the reader transparently falls back to the sequential {@link FrameReaderDelta}.
 * </p>
 */
public class FrameReaderDeltaParallel extends FrameReaderDelta {

	private final int _numThreads;

	public FrameReaderDeltaParallel() {
		_numThreads = OptimizerUtils.getParallelBinaryReadParallelism();
	}

	@Override
	public FrameBlock readFrameFromHDFS(String fname, ValueType[] schema, String[] names, long rlen, long clen)
		throws IOException, DMLRuntimeException {
		Engine engine = DeltaKernelUtils.createEngine();
		String tablePath = DeltaKernelUtils.qualify(fname);
		DeltaKernelUtils.ScanHandle handle = DeltaKernelUtils.openScan(engine, tablePath);

		final int nfiles = handle.scanFiles.size();
		// nothing to gain from parallelism for single-file (or empty) tables
		if(_numThreads <= 1 || nfiles <= 1)
			return super.readFrameFromHDFS(fname, schema, names, rlen, clen);

		// derive per-column read codes, value types and names once from the schema
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

		// fast path: exact per-file row counts are known from metadata -> pre-size
		// one typed array per column and let each thread decode directly into its
		// row offset (no intermediate buffers, no serial concatenation).
		if(useDirectPath(handle)) {
			long total = 0;
			for(long r : handle.numRecords)
				total += r;
			if(total > 0 && total <= Integer.MAX_VALUE)
				return readDirect(fname, handle, ncol, readCodes, vt, cnames, (int) total);
		}

		return readBuffered(fname, handle, ncol, readCodes, vt, cnames);
	}

	/**
	 * Fast path: each thread decodes one data file straight into the final typed column arrays at a metadata-derived
	 * row offset. Single allocation per column, fully parallel.
	 */
	private FrameBlock readDirect(String fname, DeltaKernelUtils.ScanHandle handle, int ncol, int[] readCodes,
		ValueType[] vt, String[] cnames, int nrow) throws IOException {
		final int nfiles = handle.scanFiles.size();
		final int[] rowOffset = new int[nfiles];
		int acc = 0;
		for(int i = 0; i < nfiles; i++) {
			rowOffset[i] = acc;
			acc += (int) handle.numRecords[i];
		}

		// pre-size one typed array per column for the whole table
		final Object[] dest = new Object[ncol];
		for(int c = 0; c < ncol; c++)
			dest[c] = ArrayFactory.allocateBacking(vt[c], nrow);

		ArrayList<Callable<Object>> tasks = new ArrayList<>(nfiles);
		for(int i = 0; i < nfiles; i++) {
			final Row scanFileRow = handle.scanFiles.get(i);
			final int base = rowOffset[i];
			// exclusive upper row bound for this file's slice; a file decoding more
			// rows than its numRecords statistic would otherwise overflow into the
			// next file's region (concurrent overlapping writes) or off the array
			final int limit = base + (int) handle.numRecords[i];
			tasks.add(() -> {
				int[] cur = new int[] {base};
				Engine eng = DeltaKernelUtils.createEngine();
				DeltaKernelUtils.readScanFile(eng, handle.scanState, handle.physicalReadSchema, scanFileRow,
					(cols, size, selected) -> {
						if(cur[0] + DeltaKernelUtils.countSelected(size, selected) > limit)
							throw new DMLRuntimeException("Delta file produced more rows than its "
								+ "numRecords statistic; refusing parallel direct read of " + fname);
						for(int c = 0; c < ncol; c++)
							extractColumnInto(cols[c], size, selected, readCodes[c], dest[c], cur[0]);
						cur[0] += DeltaKernelUtils.countSelected(size, selected);
					});
				return null;
			});
		}
		awaitFileTasks(tasks, fname);

		Array<?>[] columns = new Array<?>[ncol];
		for(int c = 0; c < ncol; c++)
			columns[c] = ArrayFactory.create(vt[c], dest[c]);

		FrameBlock ret = new FrameBlock(columns);
		ret.setColumnNames(cnames);
		return ret;
	}

	/**
	 * Fallback path: decode each file in parallel into per-file per-column batch arrays (used when row counts are
	 * unknown or deletion vectors are present), then concatenate per column in file order via the shared
	 * {@link FrameReaderDelta#concatColumn} helper.
	 */
	private FrameBlock readBuffered(String fname, DeltaKernelUtils.ScanHandle handle, int ncol, int[] readCodes,
		ValueType[] vt, String[] cnames) throws IOException {
		final int nfiles = handle.scanFiles.size();
		@SuppressWarnings("unchecked")
		final ArrayList<Object[]>[] fileCols = new ArrayList[nfiles];
		@SuppressWarnings("unchecked")
		final ArrayList<Integer>[] fileSizes = new ArrayList[nfiles];
		ArrayList<Callable<Object>> tasks = new ArrayList<>(nfiles);
		for(int i = 0; i < nfiles; i++) {
			final int fi = i;
			final Row scanFileRow = handle.scanFiles.get(i);
			tasks.add(() -> {
				ArrayList<Object[]> fileBatchCols = new ArrayList<>();
				ArrayList<Integer> fileBatchSizes = new ArrayList<>();
				Engine eng = DeltaKernelUtils.createEngine();
				DeltaKernelUtils.readScanFile(eng, handle.scanState, handle.physicalReadSchema, scanFileRow,
					(cols, size, selected) -> {
						int n = DeltaKernelUtils.countSelected(size, selected);
						Object[] extracted = new Object[ncol];
						for(int c = 0; c < ncol; c++) {
							// decode into a fresh per-batch array via the shared alloc +
							// decode primitives (the same ones the direct path uses)
							Object col = ArrayFactory.allocateBacking(vt[c], n);
							extractColumnInto(cols[c], size, selected, readCodes[c], col, 0);
							extracted[c] = col;
						}
						fileBatchCols.add(extracted);
						fileBatchSizes.add(n);
					});
				fileCols[fi] = fileBatchCols;
				fileSizes[fi] = fileBatchSizes;
				return null;
			});
		}
		awaitFileTasks(tasks, fname);

		// flatten the per-file batches in file order and concatenate per column
		ArrayList<Object[]> batchCols = new ArrayList<>();
		ArrayList<Integer> batchSizes = new ArrayList<>();
		int nrow = 0;
		for(int i = 0; i < nfiles; i++) {
			batchCols.addAll(fileCols[i]);
			batchSizes.addAll(fileSizes[i]);
			for(int n : fileSizes[i])
				nrow += n;
		}

		Array<?>[] columns = new Array<?>[ncol];
		for(int c = 0; c < ncol; c++)
			columns[c] = concatColumn(vt[c], nrow, batchCols, batchSizes, c);

		FrameBlock ret = new FrameBlock(columns);
		ret.setColumnNames(cnames);
		return ret;
	}

	/**
	 * Run one decode task per data file on the shared common thread pool and await completion. Full parallelism is
	 * requested (the task count, one per data file, naturally caps concurrency); this avoids the per-thread pool-size
	 * caching in {@code CommonThreadPool.get(k)} that could otherwise throttle this reader to a smaller pool created
	 * earlier on the same thread.
	 */
	private void awaitFileTasks(List<Callable<Object>> tasks, String fname) throws IOException {
		ExecutorService pool = CommonThreadPool.get(_numThreads);
		try {
			for(Future<Object> f : pool.invokeAll(tasks))
				f.get();
		}
		catch(InterruptedException ex) {
			Thread.currentThread().interrupt();
			throw new IOException("Interrupted during parallel read of Delta table: " + fname, ex);
		}
		catch(Exception ex) {
			throw new IOException("Failed parallel read of Delta table: " + fname, ex);
		}
		finally {
			pool.shutdown();
		}
	}

}
