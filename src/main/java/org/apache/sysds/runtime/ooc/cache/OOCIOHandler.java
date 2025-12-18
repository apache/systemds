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

package org.apache.sysds.runtime.ooc.cache;

import java.util.concurrent.CompletableFuture;
import java.util.List;

public interface OOCIOHandler {
	void shutdown();

	CompletableFuture<Void> scheduleEviction(BlockEntry block);

	CompletableFuture<BlockEntry> scheduleRead(BlockEntry block);

	CompletableFuture<Boolean> scheduleDeletion(BlockEntry block);

	/**
	 * Registers the source location of a block for future direct reads.
	 */
	void registerSourceLocation(BlockKey key, SourceBlockDescriptor descriptor);

	/**
	 * Schedule an asynchronous read from an external source into the provided target stream.
	 * The returned future completes when either EOF is reached or the requested byte budget
	 * is exhausted. When the budget is reached and keepOpenOnLimit is true, the target stream
	 * is kept open and a continuation token is provided so the caller can resume.
	 */
	CompletableFuture<SourceReadResult> scheduleSourceRead(SourceReadRequest request);

	/**
	 * Continue a previously throttled source read using the provided continuation token.
	 */
	CompletableFuture<SourceReadResult> continueSourceRead(SourceReadContinuation continuation, long maxBytesInFlight);

	interface SourceReadContinuation {}

	class SourceReadRequest {
		public final String path;
		public final org.apache.sysds.common.Types.FileFormat format;
		public final long rows;
		public final long cols;
		public final int blen;
		public final long estNnz;
		public final long maxBytesInFlight;
		public final boolean keepOpenOnLimit;
		public final org.apache.sysds.runtime.instructions.ooc.OOCStream<org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue> target;

		public SourceReadRequest(String path, org.apache.sysds.common.Types.FileFormat format, long rows, long cols,
			int blen, long estNnz, long maxBytesInFlight, boolean keepOpenOnLimit,
			org.apache.sysds.runtime.instructions.ooc.OOCStream<org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue> target) {
			this.path = path;
			this.format = format;
			this.rows = rows;
			this.cols = cols;
			this.blen = blen;
			this.estNnz = estNnz;
			this.maxBytesInFlight = maxBytesInFlight;
			this.keepOpenOnLimit = keepOpenOnLimit;
			this.target = target;
		}
	}

	class SourceReadResult {
		public final long bytesRead;
		public final boolean eof;
		public final SourceReadContinuation continuation;
		public final List<SourceBlockDescriptor> blocks;

		public SourceReadResult(long bytesRead, boolean eof, SourceReadContinuation continuation,
			List<SourceBlockDescriptor> blocks) {
			this.bytesRead = bytesRead;
			this.eof = eof;
			this.continuation = continuation;
			this.blocks = blocks;
		}
	}

	class SourceBlockDescriptor {
		public final String path;
		public final org.apache.sysds.common.Types.FileFormat format;
		public final org.apache.sysds.runtime.matrix.data.MatrixIndexes indexes;
		public final long offset;
		public final int recordLength;
		public final long serializedSize;

		public SourceBlockDescriptor(String path, org.apache.sysds.common.Types.FileFormat format,
			org.apache.sysds.runtime.matrix.data.MatrixIndexes indexes, long offset, int recordLength,
			long serializedSize) {
			this.path = path;
			this.format = format;
			this.indexes = indexes;
			this.offset = offset;
			this.recordLength = recordLength;
			this.serializedSize = serializedSize;
		}
	}
}
