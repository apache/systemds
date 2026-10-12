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

import java.io.IOException;
import java.io.Serializable;

import org.apache.sysds.runtime.lineage.LineageCacheConfig.ReuseCacheType;

import io.netty.buffer.ByteBuf;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.ChannelOutboundHandlerAdapter;
import io.netty.channel.ChannelPromise;
import io.netty.handler.stream.ChunkedWriteHandler;

public class FederatedFormatEncoder extends ChannelOutboundHandlerAdapter {
	private final int _chunkSize;
	private final long _streamThreshold;

	public FederatedFormatEncoder() {
		this(FederatedChunkProtocol.DEFAULT_CHUNK_SIZE, FederatedChunkProtocol.STREAM_THRESHOLD);
	}

	public FederatedFormatEncoder(int chunkSize, long streamThreshold) {
		_chunkSize = chunkSize;
		_streamThreshold = streamThreshold;
	}

	@Override
	public void write(ChannelHandlerContext ctx, Object msg, ChannelPromise promise) throws IOException {
		if(!(msg instanceof Serializable)) {
			throw new IOException("Network message must be serializable.");
		}
		if(useObjectEncoder(msg)) {
			ctx.write(markerBuffer(ctx, FederatedChunkProtocol.MARKER_OBJECT_ENCODER), ctx.voidPromise());
			ctx.write(msg, promise);
		}
		else {
			ctx.write(markerBuffer(ctx, FederatedChunkProtocol.MARKER_CHUNKED), ctx.voidPromise());
			ctx.write(FederatedChunkEncoder.chunkedInput((Serializable) msg, _chunkSize, ctx.alloc(),
				ctx.pipeline().get(ChunkedWriteHandler.class)), promise);
		}
	}

	/**
	 * Check whether a message must take the object encoder path.
	 *
	 * @param msg outbound message
	 * @return true if lineage reuse is enabled and the message is a reusable response below the stream threshold
	 */
	private boolean useObjectEncoder(Object msg) {
		if(ReuseCacheType.isNone() || !(msg instanceof FederatedResponse))
			return false;
		FederatedResponse resp = (FederatedResponse) msg;
		// Chunks are not supported by the lineage cache yet; use regular object encoder in this case if
		// the response size does not exceed the upper limits of the object encoder
		return resp.isLineageReusable() && resp.estimateSerializationBufferSize() < _streamThreshold;
	}

	/**
	 * Create a one byte buffer holding the given format marker.
	 *
	 * @param ctx  handler context supplying the allocator
	 * @param type marker byte
	 * @return buffer holding the marker
	 */
	private static ByteBuf markerBuffer(ChannelHandlerContext ctx, byte type) {
		return ctx.alloc().buffer(1).writeByte(type);
	}
}
