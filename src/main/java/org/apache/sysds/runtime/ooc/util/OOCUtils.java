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

package org.apache.sysds.runtime.ooc.util;

import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.instructions.ooc.OOCStream;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.ooc.cache.BlockEntry;
import org.apache.sysds.runtime.ooc.cache.OOCCache;
import org.apache.sysds.runtime.ooc.cache.OOCFuture;
import org.apache.sysds.runtime.ooc.memory.InMemoryQueueCallback;
import org.apache.sysds.runtime.ooc.memory.MemoryAllowance;
import org.apache.sysds.runtime.ooc.memory.ReservationBudget;
import org.apache.sysds.runtime.ooc.planning.OOCAccessPattern;
import org.apache.sysds.runtime.util.IndexRange;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.function.BooleanSupplier;

public class OOCUtils {
	public static OOCFuture<BlockEntry> pinAdmitted(OOCCache cache, long streamId, long sequenceNumber,
		MemoryAllowance allowance, BooleanSupplier cancelled) {
		if(cancelled.getAsBoolean())
			return OOCFuture.completed(null);
		if(allowance.isShutdown())
			return OOCFuture
				.failed(new IllegalStateException("Allowance was shut down while a pin admission was pending."));

		OOCFuture<BlockEntry> admitted;
		try {
			admitted = cache.pinAdmitted(streamId, sequenceNumber, allowance);
		}
		catch(RuntimeException ex) {
			return OOCFuture.failed(ex);
		}
		OOCFuture<BlockEntry> result = new OOCFuture<>();
		admitted.whenComplete((entry, error) -> {
			if(error != null) {
				result.completeExceptionally(error);
				return;
			}
			if(entry != null && cancelled.getAsBoolean()) {
				cache.unpin(entry, allowance);
				result.complete(null);
				return;
			}
			result.complete(entry);
		});
		return result;
	}

	public static IndexRange getRangeOfTile(MatrixIndexes tileIdx, long blen) {
		long rs = 1 + tileIdx.getRowIndex() * blen;
		long re = (tileIdx.getRowIndex() + 1) * blen;
		long cs = 1 + tileIdx.getColumnIndex() * blen;
		long ce = (tileIdx.getColumnIndex() + 1) * blen;
		return new IndexRange(rs, re, cs, ce);
	}

	public static Collection<MatrixIndexes> getTilesOfRange(IndexRange range, long blen) {
		long rs = (range.rowStart - 1) / blen + 1;
		long re = (range.rowEnd - 1) / blen + 1;
		long cs = (range.colStart - 1) / blen + 1;
		long ce = (range.colEnd - 1) / blen + 1;

		if(rs == re) {
			if(cs == ce) {
				return Collections.singleton(new MatrixIndexes(rs, cs));
			}
			else {
				List<MatrixIndexes> list = new ArrayList<>((int)(ce-cs+1));
				for(long i = cs; i <= ce; i++)
					list.add(new MatrixIndexes(rs, i));
				return list;
			}
		}

		List<MatrixIndexes> list = new ArrayList<>((int)((re-rs+1)*(ce-cs+1)));
		for(long r = rs; r <= re; r++)
			for (long c = cs; c <= ce; c++)
				list.add(new MatrixIndexes(r, c));
		return list;
	}

	public static long getNumBlocks(DataCharacteristics dc) {
		if (dc != null && dc.dimsKnown() && dc.getBlocksize() > 0) {
			if(dc.getCols() == 0 || dc.getRows() == 0)
				return 0;
			return dc.getNumBlocks();
		}
		return -1;
	}

	public static Iterable<MatrixIndexes> getAccessPattern(DataCharacteristics dc, OOCAccessPattern pattern) {
		long rows = dc.getRows() == 0 ? 0 : dc.getNumRowBlocks();
		long cols = dc.getCols() == 0 ? 0 : dc.getNumColBlocks();
		return getAccessPattern(rows, cols, pattern);
	}

	public static Iterable<MatrixIndexes> getAccessPattern(long rows, long cols, OOCAccessPattern pattern) {
		return () -> new Iterator<>() {
			private final long _size = rows * cols;
			private long _position;

			@Override
			public boolean hasNext() {
				return _position < _size;
			}

			@Override
			public MatrixIndexes next() {
				long position = _position++;
				return pattern == OOCAccessPattern.COL_MAJOR ? new MatrixIndexes(position % rows + 1,
					position / rows + 1) : new MatrixIndexes(position / cols + 1, position % cols + 1);
			}
		};
	}

	public static long estimateOutputTileBytes(DataCharacteristics dc) {
		if(dc == null || dc.getBlocksize() <= 0 || !dc.dimsKnown()) {
			int blocksize = dc != null && dc.getBlocksize() > 0 ? dc.getBlocksize() : 1000;
			return estimateMatrixBlockBytes(blocksize, blocksize);
		}
		return estimateMatrixBlockBytes(Math.min(dc.getBlocksize(), dc.getRows()),
			Math.min(dc.getBlocksize(), dc.getCols()));
	}

	private static long estimateMatrixBlockBytes(long rows, long cols) {
		return Math.max(MatrixBlock.estimateSizeDenseInMemory(rows, cols),
			MatrixBlock.estimateSizeSparseInMemory(rows, cols, 1.0));
	}

	public static void enqueueExact(OOCStream<IndexedMatrixValue> out, IndexedMatrixValue value,
		ReservationBudget budget) {
		long bytes = ((MatrixBlock) value.getValue()).getExactSerializedSize();
		OOCStream.QueueCallback<IndexedMatrixValue> callback = null;
		try {
			budget.reserveBlocking(bytes);
			callback = new InMemoryQueueCallback(value, null, budget, bytes);
			budget.close();
			out.enqueue(callback);
			callback = null;
		}
		finally {
			budget.close();
			if(callback != null)
				callback.close();
		}
	}
}
