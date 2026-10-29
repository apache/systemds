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

public final class FederatedChunkProtocol {
	public static final byte TYPE_DATA = 0;
	public static final byte TYPE_END = 1;
	public static final byte TYPE_ERROR = 2;

	public static final byte MARKER_OBJECT_ENCODER = 0;
	public static final byte MARKER_CHUNKED = 1;

	public static final long STREAM_THRESHOLD = 2000L << 20;

	public static final int HEADER_LEN = 5;
	public static final int DEFAULT_CHUNK_SIZE = 1 << 22;
	public static final int QUEUE_DEPTH = 16;

	public static final int LENGTH_FIELD_OFFSET = 1;
	public static final int LENGTH_FIELD_LENGTH = 4;
	public static final int LENGTH_ADJUSTMENT = 0;
	public static final int INITIAL_BYTES_TO_STRIP = 0;

	/**
	 * Get the largest frame a chunk of the given size can produce.
	 *
	 * @param chunkSize payload bytes per data frame
	 * @return frame size including the header
	 */
	static int maxChunkFrameLength(int chunkSize) {
		return chunkSize + HEADER_LEN;
	}

	/**
	 * Create a length based frame decoder for the chunked wire format.
	 *
	 * @return frame decoder sized for the default chunk size
	 */
	static LengthFieldBasedFrameDecoder newChunkFrameDecoder() {
		return new LengthFieldBasedFrameDecoder(maxChunkFrameLength(DEFAULT_CHUNK_SIZE), LENGTH_FIELD_OFFSET,
			LENGTH_FIELD_LENGTH, LENGTH_ADJUSTMENT, INITIAL_BYTES_TO_STRIP);
	}

	private FederatedChunkProtocol() {
	}
}
