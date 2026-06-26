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

import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.CommonThreadPool;

import io.delta.kernel.data.Row;
import io.delta.kernel.engine.Engine;

/**
 * Parallel native Delta Lake matrix reader. Delta tables are stored as one or
 * more parquet data files; this reader decodes those files concurrently (one
 * task per data file) and then concatenates the per-file row-major buffers into
 * the dense output in the original file order.
 *
 * <p>The expensive part of a Delta read is the parquet decode, which the kernel
 * performs per data file; parallelizing across files is therefore the natural
 * way to bridge the gap to the (near-raw) binary reader. A table backed by a
 * single data file (the default for tables &lt;= the parquet target file size)
 * cannot be split this way, so the reader transparently falls back to the
 * sequential {@link ReaderDelta} path in that case.</p>
 */
public class ReaderDeltaParallel extends ReaderDelta {

	private final int _numThreads;

	public ReaderDeltaParallel() {
		_numThreads = OptimizerUtils.getParallelBinaryReadParallelism();
	}

	@Override
	public MatrixBlock readMatrixFromHDFS(String fname, long rlen, long clen, int blen, long estnnz)
		throws IOException, DMLRuntimeException
	{
		Engine engine = DeltaKernelUtils.createEngine();
		String tablePath = DeltaKernelUtils.qualify(fname);
		DeltaKernelUtils.ScanHandle handle = DeltaKernelUtils.openScan(engine, tablePath);

		final int nfiles = handle.scanFiles.size();
		//nothing to gain from parallelism for single-file (or empty) tables
		if( _numThreads <= 1 || nfiles <= 1 )
			return super.readMatrixFromHDFS(fname, rlen, clen, blen, estnnz);

		final int ncol = handle.schema.length();
		final int[] types = columnTypes(handle.schema);

		//fast path: exact per-file row counts are known from metadata and the dense
		//output fits a single contiguous array -> pre-size once and let each thread
		//decode directly into its slice (no intermediate buffers, no serial copy).
		if( useDirectPath(handle) ) {
			long total = 0;
			for( long r : handle.numRecords )
				total += r;
			if( total > 0 && (long) total * ncol <= Integer.MAX_VALUE )
				return readDirect(fname, handle, ncol, types, (int) total, estnnz);
		}

		return readBuffered(fname, handle, ncol, types, estnnz);
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
	 * Fast path: each thread decodes one data file straight into the final dense
	 * array at a metadata-derived row offset. Single allocation, fully parallel.
	 */
	private MatrixBlock readDirect(String fname, DeltaKernelUtils.ScanHandle handle,
		int ncol, int[] types, int nrow, long estnnz) throws IOException
	{
		final int nfiles = handle.scanFiles.size();
		final int[] rowOffset = new int[nfiles];
		int acc = 0;
		for( int i=0; i<nfiles; i++ ) {
			rowOffset[i] = acc;
			acc += (int) handle.numRecords[i];
		}

		//force a contiguous dense allocation (matrices from Delta are dense doubles)
		MatrixBlock ret = createOutputMatrixBlock(nrow, ncol, Math.max(nrow, 1), (long) nrow * ncol, true, false);
		final double[] dv = ret.getDenseBlock().valuesAt(0);

		ArrayList<Callable<Object>> tasks = new ArrayList<>(nfiles);
		for( int i=0; i<nfiles; i++ ) {
			final Row scanFileRow = handle.scanFiles.get(i);
			final int base = rowOffset[i];
			tasks.add(() -> {
				int[] cur = new int[] {base};
				Engine eng = DeltaKernelUtils.createEngine();
				DeltaKernelUtils.readScanFile(eng, handle.scanState, handle.physicalReadSchema, scanFileRow,
					(cols, size, selected) ->
						cur[0] += extractBatchInto(cols, size, selected, types, ncol, dv, cur[0]));
				return null;
			});
		}
		awaitFileTasks(tasks, fname);

		ret.recomputeNonZeros(_numThreads);
		ret.examSparsity();
		return ret;
	}

	/**
	 * Fallback path: decode each file in parallel into per-file buffers (used when
	 * row counts are unknown, deletion vectors are present, or the matrix exceeds a
	 * single contiguous array), then concatenate in file order.
	 */
	private MatrixBlock readBuffered(String fname, DeltaKernelUtils.ScanHandle handle,
		int ncol, int[] types, long estnnz) throws IOException
	{
		final int nfiles = handle.scanFiles.size();
		@SuppressWarnings("unchecked")
		final ArrayList<double[]>[] fileBufs = new ArrayList[nfiles];
		final int[] fileRows = new int[nfiles];
		ArrayList<Callable<Object>> tasks = new ArrayList<>(nfiles);
		for( int i=0; i<nfiles; i++ ) {
			final int fi = i;
			final Row scanFileRow = handle.scanFiles.get(i);
			tasks.add(() -> {
				ArrayList<double[]> bufs = new ArrayList<>();
				int[] rows = new int[1];
				Engine eng = DeltaKernelUtils.createEngine();
				DeltaKernelUtils.readScanFile(eng, handle.scanState, handle.physicalReadSchema, scanFileRow,
					(cols, size, selected) -> {
						bufs.add(extractBatch(cols, size, selected, types, ncol));
						rows[0] += DeltaKernelUtils.countSelected(size, selected);
					});
				fileBufs[fi] = bufs;
				fileRows[fi] = rows[0];
				return null;
			});
		}
		awaitFileTasks(tasks, fname);

		int nrow = 0;
		for( int i=0; i<nfiles; i++ )
			nrow += fileRows[i];
		ArrayList<double[]> ordered = new ArrayList<>();
		for( int i=0; i<nfiles; i++ )
			ordered.addAll(fileBufs[i]);

		long lestnnz = (estnnz >= 0) ? estnnz : (long) nrow * ncol;
		MatrixBlock ret = createOutputMatrixBlock(nrow, ncol, Math.max(nrow, 1), lestnnz, true, false);
		if( nrow > 0 && ncol > 0 )
			fillDense(ret, ordered);
		ret.recomputeNonZeros();
		ret.examSparsity();
		return ret;
	}

	/**
	 * Run one decode task per data file on the shared common thread pool and await
	 * completion. Full parallelism is requested (the task count, one per data file,
	 * naturally caps concurrency); this avoids the per-thread pool-size caching in
	 * {@code CommonThreadPool.get(k)} that could otherwise throttle this reader to a
	 * smaller pool created earlier on the same thread.
	 */
	private void awaitFileTasks(List<Callable<Object>> tasks, String fname) throws IOException {
		ExecutorService pool = CommonThreadPool.get(_numThreads);
		try {
			for( Future<Object> f : pool.invokeAll(tasks) )
				f.get();
		}
		catch(Exception ex) {
			throw new IOException("Failed parallel read of Delta table: " + fname, ex);
		}
		finally {
			pool.shutdown();
		}
	}
}
