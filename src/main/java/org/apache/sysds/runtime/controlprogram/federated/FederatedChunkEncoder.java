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
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.io.Serializable;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;

import org.apache.sysds.runtime.util.CommonThreadPool;

import io.netty.buffer.ByteBuf;
import io.netty.buffer.ByteBufAllocator;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.ChannelOutboundHandlerAdapter;
import io.netty.channel.ChannelPromise;
import io.netty.handler.stream.ChunkedInput;
import io.netty.handler.stream.ChunkedWriteHandler;

public class FederatedChunkEncoder extends ChannelOutboundHandlerAdapter {
	private final int _chunkSize;

	public FederatedChunkEncoder() {
		this(FederatedChunkProtocol.DEFAULT_CHUNK_SIZE);
	}

	public FederatedChunkEncoder(int chunkSize) {
		_chunkSize = chunkSize;
	}

	static ChunkedInput<ByteBuf> chunkedInput(Serializable msg, int chunkSize, ByteBufAllocator alloc,
		ChunkedWriteHandler writer) {
		return new SerializedChunks(msg, chunkSize, alloc, writer);
	}

	@Override
	public void write(ChannelHandlerContext ctx, Object msg, ChannelPromise promise) {
		if(msg instanceof Serializable)
			ctx.write(new SerializedChunks((Serializable) msg, _chunkSize, ctx.alloc(),
				ctx.pipeline().get(ChunkedWriteHandler.class)), promise);
		else
			ctx.write(msg, promise);
	}

	private static final class SerializedChunks implements ChunkedInput<ByteBuf> {
		private final BlockingQueue<ByteBuf> _frames = new ArrayBlockingQueue<>(FederatedChunkProtocol.QUEUE_DEPTH);
		private final ByteBufAllocator _alloc;
		private final ChunkedWriteHandler _writer;
		private volatile boolean _closed;
		private boolean _done;

		SerializedChunks(Serializable msg, int chunkSize, ByteBufAllocator alloc, ChunkedWriteHandler writer) {
			_alloc = alloc;
			_writer = writer;
			CommonThreadPool.getDynamicPool().execute(() -> produceFrames(msg, chunkSize));
		}

		private void produceFrames(Serializable msg, int chunkSize) {
			try(FrameOutputStream out = new FrameOutputStream(this, _alloc, chunkSize);
				ObjectOutputStream oos = new ObjectOutputStream(out)) {
				oos.writeObject(msg);
				oos.flush();
				out.flushFrame();
				enqueueControlFrame(controlFrame(FederatedChunkProtocol.TYPE_END));
			}
			catch(Throwable t) {
				enqueueControlFrame(errorFrame(t));
			}
		}

		private ByteBuf controlFrame(byte type) {
			return _alloc.buffer(FederatedChunkProtocol.HEADER_LEN).writeByte(type).writeInt(0);
		}

		private ByteBuf errorFrame(Throwable t) {
			byte[] cause = String.valueOf(t).getBytes(StandardCharsets.UTF_8);
			return _alloc.buffer(FederatedChunkProtocol.HEADER_LEN + cause.length)
				.writeByte(FederatedChunkProtocol.TYPE_ERROR).writeInt(cause.length).writeBytes(cause);
		}

		void enqueueFrame(ByteBuf frame) throws InterruptedException {
			if(_closed) {
				frame.release();
				return;
			}
			_frames.put(frame);
			_writer.resumeTransfer();
		}

		private void enqueueControlFrame(ByteBuf frame) {
			try {
				enqueueFrame(frame);
			}
			catch(InterruptedException e) {
				frame.release();
				Thread.currentThread().interrupt();
			}
		}

		@Override
		public ByteBuf readChunk(ByteBufAllocator allocator) {
			if(_done)
				return null;
			ByteBuf frame = _frames.poll();
			if(frame == null)
				return null;
			_done = frame.getByte(frame.readerIndex()) != FederatedChunkProtocol.TYPE_DATA;
			return frame;
		}

		@Override
		public ByteBuf readChunk(ChannelHandlerContext ctx) {
			return readChunk(ctx.alloc());
		}

		@Override
		public boolean isEndOfInput() {
			return _done;
		}

		@Override
		public long length() {
			return -1;
		}

		@Override
		public long progress() {
			return 0;
		}

		@Override
		public void close() {
			_closed = true;
			ByteBuf frame;
			while((frame = _frames.poll()) != null)
				frame.release();
		}
	}

	private static final class FrameOutputStream extends OutputStream {
		private final SerializedChunks _sink;
		private final ByteBufAllocator _alloc;
		private final byte[] _buffer;
		private int _len;

		FrameOutputStream(SerializedChunks sink, ByteBufAllocator alloc, int chunkSize) {
			_sink = sink;
			_alloc = alloc;
			_buffer = new byte[chunkSize];
		}

		@Override
		public void write(int b) throws IOException {
			_buffer[_len++] = (byte) b;
			if(_len == _buffer.length)
				flushFrame();
		}

		@Override
		public void write(byte[] b, int off, int len) throws IOException {
			while(len > 0) {
				int n = Math.min(len, _buffer.length - _len);
				System.arraycopy(b, off, _buffer, _len, n);
				_len += n;
				off += n;
				len -= n;
				if(_len == _buffer.length)
					flushFrame();
			}
		}

		void flushFrame() throws IOException {
			if(_len == 0)
				return;
			ByteBuf frame = _alloc.buffer(FederatedChunkProtocol.HEADER_LEN + _len)
				.writeByte(FederatedChunkProtocol.TYPE_DATA).writeInt(_len).writeBytes(_buffer, 0, _len);
			_len = 0;
			try {
				_sink.enqueueFrame(frame);
			}
			catch(InterruptedException e) {
				frame.release();
				Thread.currentThread().interrupt();
				throw new IOException(e);
			}
		}
	}
}
