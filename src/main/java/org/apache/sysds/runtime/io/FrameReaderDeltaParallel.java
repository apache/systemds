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
		// nothing to gain from parallelism for single-file (or empty) tables: reuse
		// the already-opened engine + scan handle instead of re-opening the snapshot.
		if(_numThreads <= 1 || nfiles <= 1)
			return readWithHandle(fname, engine, handle);

		// derive per-column read codes, value types and names once from the schema
		final ReadPlan plan = planColumns(handle);

		// fast path: exact per-file row counts are known from metadata -> pre-size
		// one typed array per column and let each thread decode directly into its
		// row offset (no intermediate buffers, no serial concatenation).
		if(useDirectPath(handle)) {
			long total = 0;
			for(long r : handle.numRecords)
				total += r;
			if(total > 0 && total <= Integer.MAX_VALUE)
				return readDirect(fname, handle, plan, (int) total);
		}

		return readBuffered(fname, handle, plan);
	}

	/**
	 * Fast path: each thread decodes one data file straight into the final typed column arrays at a metadata-derived
	 * row offset, through parquet-mr's column API with no kernel engine in the path (and hence no per-file engine
	 * creation). Single allocation per column, fully parallel.
	 */
	private FrameBlock readDirect(String fname, DeltaKernelUtils.ScanHandle handle, ReadPlan plan, int nrow)
		throws IOException {
		final int ncol = plan.ncol;
		final int[] readCodes = plan.readCodes;
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
			dest[c] = ArrayFactory.allocateBacking(plan.vt[c], nrow);

		ArrayList<Callable<Object>> tasks = new ArrayList<>(nfiles);
		for(int i = 0; i < nfiles; i++) {
			final Row scanFileRow = handle.scanFiles.get(i);
			final int base = rowOffset[i];
			// exclusive upper row bound for this file's slice; enforced inside the
			// decode before each row group is written, so a file with more rows than
			// its numRecords statistic cannot overflow into the next file's region
			final int limit = base + (int) handle.numRecords[i];
			tasks.add(() -> {
				String path = DeltaKernelUtils.dataFilePath(scanFileRow);
				int n = DeltaKernelUtils.decodeDataFileInto(path, handle.physicalReadSchema, readCodes, dest, base,
					limit, fname);
				// fail loud on underflow too: fewer decoded rows than the statistic
				// would leave this slice's tail at the array default (0/null).
				if(base + n != limit)
					throw new DMLRuntimeException("Delta file produced " + n + " rows, expected " + (limit - base)
						+ " from its numRecords statistic; refusing parallel direct read of " + fname);
				return null;
			});
		}
		awaitFileTasks(tasks, fname);

		Array<?>[] columns = new Array<?>[ncol];
		for(int c = 0; c < ncol; c++)
			columns[c] = ArrayFactory.create(plan.vt[c], dest[c]);

		FrameBlock ret = new FrameBlock(columns);
		ret.setColumnNames(plan.cnames);
		return ret;
	}

	/**
	 * Fallback path: decode each file in parallel into per-file per-column batch arrays (used when row counts are
	 * unknown or deletion vectors are present), then concatenate per column in file order via the shared
	 * {@link FrameReaderDelta#concatColumn} helper.
	 */
	private FrameBlock readBuffered(String fname, DeltaKernelUtils.ScanHandle handle, ReadPlan plan)
		throws IOException {
		final int ncol = plan.ncol;
		final int[] readCodes = plan.readCodes;
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
							Object col = ArrayFactory.allocateBacking(plan.vt[c], n);
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
			columns[c] = concatColumn(plan.vt[c], nrow, batchCols, batchSizes, c);

		FrameBlock ret = new FrameBlock(columns);
		ret.setColumnNames(plan.cnames);
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
