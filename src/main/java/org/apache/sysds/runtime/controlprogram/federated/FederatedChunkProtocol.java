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

package org.apache.sysds.runtime.controlprogram.federated;

import io.netty.handler.codec.LengthFieldBasedFrameDecoder;

final class FederatedChunkProtocol {
	static final byte TYPE_DATA = 0;
	static final byte TYPE_END = 1;
	static final byte TYPE_ERROR = 2;

	static final byte MARKER_LEGACY = 0;
	static final byte MARKER_CHUNKED = 1;
	static final long STREAM_THRESHOLD = 1536L << 20; // ~1.5 GB: route below this through the legacy object codec

	static final int HEADER_LEN = 5;
	static final int DEFAULT_CHUNK_SIZE = 1 << 20; // 1 MB payload per frame
	static final int QUEUE_DEPTH = 16;

	static final int LENGTH_FIELD_OFFSET = 1;
	static final int LENGTH_FIELD_LENGTH = 4;
	static final int LENGTH_ADJUSTMENT = 0;
	static final int INITIAL_BYTES_TO_STRIP = 0;

	static int maxFrameLength(int chunkSize) {
		return chunkSize + HEADER_LEN;
	}

	static LengthFieldBasedFrameDecoder newFrameDecoder() {
		return new LengthFieldBasedFrameDecoder(maxFrameLength(DEFAULT_CHUNK_SIZE),
			LENGTH_FIELD_OFFSET, LENGTH_FIELD_LENGTH, LENGTH_ADJUSTMENT, INITIAL_BYTES_TO_STRIP);
	}

	private FederatedChunkProtocol() {}
}
