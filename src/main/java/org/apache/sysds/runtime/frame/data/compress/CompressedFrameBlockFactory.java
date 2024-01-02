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

package org.apache.sysds.runtime.frame.data.compress;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.workload.WTreeRoot;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.columns.Array;
import org.apache.sysds.runtime.frame.data.columns.ArrayFactory;
import org.apache.sysds.runtime.frame.data.columns.DDCArray;
import org.apache.sysds.runtime.util.CommonThreadPool;

public class CompressedFrameBlockFactory {

	private static final Log LOG = LogFactory.getLog(CompressedFrameBlockFactory.class.getName());

	private final FrameBlock in;
	private final FrameCompressionSettings cs;
	private final ArrayCompressionStatistics[] stats;
	private final Array<?>[] compressedColumns;

	private final int nSamples;

	private CompressedFrameBlockFactory(FrameBlock fb, FrameCompressionSettings cs) {
		this.in = fb;
		this.cs = cs;
		this.stats = new ArrayCompressionStatistics[in.getNumColumns()];
		this.compressedColumns = new Array<?>[in.getNumColumns()];
		this.nSamples = Math.min(in.getNumRows(), Math.max(1000, (int) Math.ceil(in.getNumRows() * cs.sampleRatio)));
	}

	public static FrameBlock compress(FrameBlock fb) {
		FrameCompressionSettings cs = new FrameCompressionSettingsBuilder().create();
		return compress(fb, cs);
	}

	public static FrameBlock compress(FrameBlock fb, int k, WTreeRoot root) {
		FrameCompressionSettings cs = new FrameCompressionSettingsBuilder()//
			.threads(k).wTreeRoot(root).create();
		return compress(fb, cs);
	}

	public static FrameBlock compress(FrameBlock fb, FrameCompressionSettingsBuilder csb) {
		return compress(fb, csb.create());
	}

	public static FrameBlock compress(FrameBlock fb, FrameCompressionSettings cs) {
		return new CompressedFrameBlockFactory(fb, cs).compressFrame();
	}

	private FrameBlock compressFrame() {
		encodeColumns();
		final FrameBlock ret = new FrameBlock(compressedColumns, in.getColumnNames(false));
		logStatistics();
		logRet(ret);
		return ret;
	}

	private void encodeColumns() {
		if(cs.k > 1)
			encodeParallel();
		else
			encodeSingleThread();
	}

	private void encodeSingleThread() {
		for(int i = 0; i < compressedColumns.length; i++)
			compressCol(i);
	}

	private void encodeParallel() {
		final ExecutorService pool = CommonThreadPool.get(cs.k);
		try {
			List<Future<?>> tasks = new ArrayList<>();
			for(int j = 0; j < compressedColumns.length; j++) {
				final int i = j;
				final Future<?> stats = pool.submit(() -> (getStatistics(i)));
				final Future<Array<?>> tmp = pool.submit(() -> allocateCorrectedType(i, stats));
				final Future<Array<?>> tmp2 = changeTypeFuture(i, tmp, pool, cs.k);
				tasks.add(pool.submit(() -> compressColFinally(i, tmp2)));
			}

			for(Future<?> t : tasks)
				t.get();

		}
		catch(Exception e) {
			throw new RuntimeException(e);
		}
		finally {
			pool.shutdown();
		}
	}

	private void compressCol(int i) {
		final ArrayCompressionStatistics s = getStatistics(i);
		if(s != null)
			compressCol(i, s);
		else
			compressedColumns[i] = in.getColumn(i);
	}

	private ArrayCompressionStatistics getStatistics(int i) {
		return stats[i] = in.getColumn(i).statistics(nSamples);
	}

	private Array<?> allocateCorrectedType(int i, Future<?> f) {
		try {
			f.get();
			return allocateCorrectedType(i);
		}
		catch(InterruptedException | ExecutionException e) {
			throw new RuntimeException(e);
		}
	}

	private void compressColFinally(int i, Future<Array<?>> f) {
		try {
			final Array<?> a = f.get();
			compressColFinally(i, a, stats[i]);
		}
		catch(InterruptedException | ExecutionException e) {
			throw new RuntimeException(e);
		}
	}

	private Array<?> allocateCorrectedType(int i) {
		final ArrayCompressionStatistics s = stats[i];
		final Array<?> a = in.getColumn(i);
		if(s.valueType != null && s.valueType != a.getValueType())
			return s.containsNull ? //
				ArrayFactory.allocateOptional(s.valueType, a.size()) : //
				ArrayFactory.allocate(s.valueType, a.size());//
		else
			return a;
	}

	private Future<Array<?>> changeTypeFuture(int i, Future<Array<?>> f, ExecutorService pool, int k) {
		try {
			final Array<?> tmp = f.get();
			final Array<?> a = in.getColumn(i);
			final ArrayCompressionStatistics s = stats[i];
			if(s.valueType != null && s.valueType != a.getValueType()) {

				final int nRow = in.getNumRows();
				final int block = Math.max(((nRow / k) / 64) * 64, 1024);

				final List<Future<?>> t = new ArrayList<>();
				for(int r = 0; r < nRow; r += block) {

					final int start = r;
					final int end = Math.min(r + block, nRow);
					t.add(pool.submit(() -> (a.changeTypeWithNulls(tmp, start, end))));
				}

				return pool.submit(() -> {
					try {
						for(Future<?> tt : t)
							tt.get();
						return tmp;
					}
					catch(Exception e) {

						throw new RuntimeException(e);
					}
				});
			}
			else
				return pool.submit(() -> tmp);

		}

		catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException(e);
		}
	}

	private void compressCol(int i, final ArrayCompressionStatistics s) {
		final Array<?> b = in.getColumn(i);
		final Array<?> a;
		if(s.valueType != null && s.valueType != b.getValueType())
			a = b.changeType(s.valueType, s.containsNull);
		else
			a = b;

		compressColFinally(i, a, s);
	}

	private void compressColFinally(int i, final Array<?> a, final ArrayCompressionStatistics s) {

		if(s.bestType != null) {
			switch(s.bestType) {
				case DDC:
					compressedColumns[i] = DDCArray.compressToDDC(a, s.containsNull);
					break;
				default:
					LOG.error("Unsupported encoding default to do nothing: " + s.bestType);
					compressedColumns[i] = a;
					break;
			}
		}
		else
			compressedColumns[i] = a;
	}

	private void logStatistics() {
		if(LOG.isDebugEnabled()) {
			for(int i = 0; i < compressedColumns.length; i++) {
				if(stats[i] != null)
					LOG.debug(String.format("Col: %3d, %s", i, stats[i]));
				else
					LOG.debug(
						String.format("Col: %3d, No Compress, Type: %s", i, in.getColumn(i).getClass().getSimpleName()));
			}
		}
	}

	private void logRet(FrameBlock ret) {
		if(LOG.isDebugEnabled()) {
			final long before = in.getInMemorySize();
			final long after = ret.getInMemorySize();
			LOG.debug(String.format("nRows              %15d", in.getNumRows()));
			LOG.debug(String.format("SampleSize         %15d", nSamples));
			LOG.debug(String.format("Uncompressed Size: %15d", before));
			LOG.debug(String.format("compressed Size:   %15d", after));
			LOG.debug(String.format("ratio:             %15.3f", (double) before / (double) after));
		}
	}

}
