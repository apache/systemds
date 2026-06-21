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
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectStreamClass;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;

import org.apache.sysds.runtime.util.CommonThreadPool;

import io.netty.buffer.ByteBuf;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.MessageToMessageDecoder;

public class FederatedChunkDecoder extends MessageToMessageDecoder<ByteBuf> {
	private static final Object END_OF_STREAM = new Object();
	// stop reading at QUEUE_DEPTH, resume at half: gap avoids autoRead thrash
	private static final int LOW_WATERMARK = FederatedChunkProtocol.QUEUE_DEPTH / 2;

	private final BlockingQueue<Object> _payloads = new LinkedBlockingQueue<>();
	private boolean _started;
	private volatile boolean _throttled;

	@Override
	protected void decode(ChannelHandlerContext ctx, ByteBuf frame, List<Object> out) {
		startReader(ctx);
		byte type = frame.readByte();
		int len = frame.readInt();
		switch(type) {
			case FederatedChunkProtocol.TYPE_DATA:
				_payloads.add(readBytes(frame, len));
				break;
			case FederatedChunkProtocol.TYPE_END:
				_payloads.add(END_OF_STREAM);
				break;
			case FederatedChunkProtocol.TYPE_ERROR:
				_payloads.add(new IOException(frame.toString(frame.readerIndex(), len, StandardCharsets.UTF_8)));
				break;
		}
		if(_payloads.size() >= FederatedChunkProtocol.QUEUE_DEPTH) {
			_throttled = true;
			ctx.channel().config().setAutoRead(false);
		}
	}

	private void startReader(ChannelHandlerContext ctx) {
		if(_started)
			return;
		_started = true;
		CommonThreadPool.getDynamicPool().execute(() -> runDeserializer(ctx));
	}

	private void runDeserializer(ChannelHandlerContext ctx) {
		try(ObjectInputStream ois = objectInputStream(new PayloadInputStream(this, ctx))) {
			Object msg = ois.readObject();
			ctx.channel().eventLoop().execute(() -> ctx.fireChannelRead(msg));
		}
		catch(Throwable t) {
			ctx.channel().eventLoop().execute(() -> ctx.fireExceptionCaught(t));
		}
	}

	private Object nextPayload() throws InterruptedException {
		return _payloads.take();
	}

	private void resumeReadingIfDrained(ChannelHandlerContext ctx) {
		if(_throttled && _payloads.size() <= LOW_WATERMARK) {
			_throttled = false;
			ctx.channel().eventLoop().execute(() -> ctx.channel().config().setAutoRead(true));
		}
	}

	private static ObjectInputStream objectInputStream(InputStream in) throws IOException {
		return new ObjectInputStream(in) {
			@Override
			protected Class<?> resolveClass(ObjectStreamClass desc) throws IOException, ClassNotFoundException {
				try {
					return Class.forName(desc.getName(), false, ClassLoader.getSystemClassLoader());
				}
				catch(ClassNotFoundException e) {
					return super.resolveClass(desc);
				}
			}
		};
	}

	private static byte[] readBytes(ByteBuf frame, int len) {
		byte[] bytes = new byte[len];
		frame.readBytes(bytes);
		return bytes;
	}

	private static final class PayloadInputStream extends InputStream {
		private static final byte[] EMPTY = new byte[0];

		private final FederatedChunkDecoder _decoder;
		private final ChannelHandlerContext _ctx;
		private byte[] _current = EMPTY;
		private int _pos;
		private boolean _eof;

		PayloadInputStream(FederatedChunkDecoder decoder, ChannelHandlerContext ctx) {
			_decoder = decoder;
			_ctx = ctx;
		}

		@Override
		public int read() throws IOException {
			if(!ensureCurrent())
				return -1;
			return _current[_pos++] & 0xff;
		}

		@Override
		public int read(byte[] b, int off, int len) throws IOException {
			if(!ensureCurrent())
				return -1;
			int n = Math.min(len, _current.length - _pos);
			System.arraycopy(_current, _pos, b, off, n);
			_pos += n;
			return n;
		}

		private boolean ensureCurrent() throws IOException {
			while(_pos == _current.length) {
				if(_eof)
					return false;
				Object next = take();
				if(next == END_OF_STREAM) {
					_eof = true;
					return false;
				}
				if(next instanceof Throwable)
					throw new IOException((Throwable) next);
				_current = (byte[]) next;
				_pos = 0;
			}
			return true;
		}

		private Object take() throws IOException {
			try {
				Object next = _decoder.nextPayload();
				_decoder.resumeReadingIfDrained(_ctx);
				return next;
			}
			catch(InterruptedException e) {
				Thread.currentThread().interrupt();
				throw new IOException(e);
			}
		}
	}
}
