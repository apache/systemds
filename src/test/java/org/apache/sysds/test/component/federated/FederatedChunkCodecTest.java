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

package org.apache.sysds.test.component.federated;

import java.io.IOException;
import java.io.Serializable;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.apache.sysds.runtime.controlprogram.federated.FederatedChunkDecoder;
import org.apache.sysds.runtime.controlprogram.federated.FederatedChunkEncoder;
import org.apache.sysds.runtime.controlprogram.federated.FederatedChunkProtocol;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse.ResponseType;
import org.junit.Assert;
import org.junit.Test;

import io.netty.buffer.AbstractByteBufAllocator;
import io.netty.buffer.ByteBuf;
import io.netty.buffer.Unpooled;
import io.netty.buffer.UnpooledByteBufAllocator;
import io.netty.channel.ChannelFuture;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.ChannelOutboundHandlerAdapter;
import io.netty.channel.ChannelPromise;
import io.netty.channel.embedded.EmbeddedChannel;
import io.netty.handler.codec.LengthFieldBasedFrameDecoder;
import io.netty.handler.codec.compression.JdkZlibDecoder;
import io.netty.handler.codec.compression.JdkZlibEncoder;
import io.netty.handler.codec.compression.ZlibWrapper;
import io.netty.handler.stream.ChunkedInput;
import io.netty.handler.stream.ChunkedWriteHandler;

public class FederatedChunkCodecTest {
	private static final int CHUNK_SIZE = 4096; // tiny on purpose: forces a multi-frame stream
	private static final int MAX_FRAME = 1 << 20; // 1 MB: frame-decoder ceiling, must exceed CHUNK_SIZE + header
	private static final int PAYLOAD_DOUBLES = 20000; // ~160 KB serialized, many CHUNK_SIZE frames
	private static final int QUEUED_DOUBLES = 2000; // ~16 KB serialized, few enough frames to never block the queue
	// mirrors the package-private FederatedChunkProtocol, which this test cannot import

	@Test
	public void roundTripPlainSplitsIntoManyFrames() throws Exception {
		FederatedResponse original = sampleResponse();
		List<ByteBuf> frames = encode(original, false);
		Assert.assertTrue("expected multiple frames", frames.size() > 2);
		assertSamePayload(original, decode(frames, false));
	}

	@Test
	public void roundTripThroughCompression() throws Exception {
		FederatedResponse original = sampleResponse();
		assertSamePayload(original, decode(encode(original, true), true));
	}

	@Test
	public void producerFailureEmitsErrorFrame() throws Exception {
		List<ByteBuf> frames = encode(new Unserializable(), false);
		Assert.assertFalse("expected an error frame", frames.isEmpty());
		ByteBuf last = frames.get(frames.size() - 1);
		Assert.assertEquals(FederatedChunkProtocol.TYPE_ERROR, last.getByte(0));
		String message = last.toString(FederatedChunkProtocol.HEADER_LEN, last.getInt(1), StandardCharsets.UTF_8);
		Assert.assertTrue(message.contains("NotSerializableException"));
		for(ByteBuf frame : frames)
			frame.release();
	}

	@Test
	public void errorFrameThrowsException() throws Exception {
		final String ERROR_MSG = "remote failure";
		byte[] cause = ERROR_MSG.getBytes(StandardCharsets.UTF_8);
		ByteBuf errorFrame = Unpooled.buffer(FederatedChunkProtocol.HEADER_LEN + cause.length)
			.writeByte(FederatedChunkProtocol.TYPE_ERROR).writeInt(cause.length).writeBytes(cause);
		Throwable caught = writeAndAwaitException(errorFrame);
		Assert.assertTrue("expected an IOException, got " + caught, caught instanceof IOException);
		Assert.assertTrue(String.valueOf(caught).contains("remote failure"));
	}

	@Test
	public void unknownFrameTypeThrowsException() throws Exception {
		byte unknownType = 7; // not part of the protocol, must fail fast instead of stalling
		ByteBuf unknownTypeFrame = Unpooled.buffer(FederatedChunkProtocol.HEADER_LEN).writeByte(unknownType)
			.writeInt(0);
		Throwable caught = writeAndAwaitException(unknownTypeFrame);
		Assert.assertTrue("expected an IOException, got " + caught, caught instanceof IOException);
		Assert.assertTrue(String.valueOf(caught).contains("Unknown federated chunk frame type: " + unknownType));
	}

	@Test
	public void closeReleasesQueuedFrames() throws Exception {
		RecordingAllocator alloc = new RecordingAllocator();
		EmbeddedChannel channel = new EmbeddedChannel(new ChunkedWriteHandler());
		ChunkedInput<ByteBuf> input = FederatedChunkEncoder.chunkedInput(sampleResponse(QUEUED_DOUBLES), CHUNK_SIZE,
			alloc, channel.pipeline().get(ChunkedWriteHandler.class));
		awaitProducerFinished(alloc);
		input.close();
		for(ByteBuf frame : alloc.frames())
			Assert.assertEquals("frame left unreleased by close()", 0, frame.refCnt());
	}

	private static FederatedResponse sampleResponse() {
		return sampleResponse(PAYLOAD_DOUBLES);
	}

	private static FederatedResponse sampleResponse(int doubles) {
		double[] data = new double[doubles];
		for(int i = 0; i < data.length; i++)
			data[i] = i;
		return new FederatedResponse(ResponseType.SUCCESS, data);
	}

	private static Throwable writeAndAwaitException(ByteBuf frame) throws InterruptedException {
		EmbeddedChannel channel = new EmbeddedChannel(frameDecoder(), new FederatedChunkDecoder());
		try {
			// the deserializer thread reports the failure asynchronously: writeInbound throws it if it already arrived,
			// otherwise awaitException below waits for it
			channel.writeInbound(frame);
		}
		catch(Throwable t) {
			return t;
		}
		return awaitException(channel);
	}

	private static Throwable awaitException(EmbeddedChannel channel) throws InterruptedException {
		for(int i = 0; i < 20; i++) {
			channel.runPendingTasks();
			try {
				channel.checkException();
			}
			catch(Throwable t) {
				return t;
			}
			Thread.sleep(50);
		}
		throw new AssertionError("no exception propagated");
	}

	private static void awaitProducerFinished(RecordingAllocator alloc) throws InterruptedException {
		for(int i = 0; i < 20; i++) {
			if(alloc.sawFinalFrame())
				return;
			Thread.sleep(50);
		}
		throw new AssertionError("producer did not finish");
	}

	private static List<ByteBuf> encode(Serializable response, boolean compress) throws Exception {
		EmbeddedChannel channel = compress ? new EmbeddedChannel(new JdkZlibEncoder(ZlibWrapper.ZLIB),
			new ChunkedWriteHandler(), chunkEncoder()) : new EmbeddedChannel(new ChunkedWriteHandler(), chunkEncoder());
		channel.config().setWriteBufferHighWaterMark(MAX_FRAME * 64);
		List<ByteBuf> frames = new ArrayList<>();
		ChannelFuture done = channel.write(response);
		channel.flush();
		pumpOutbound(channel, done, frames);
		return frames;
	}

	private static ChannelOutboundHandlerAdapter chunkEncoder() {
		return new ChannelOutboundHandlerAdapter() {
			@Override
			public void write(ChannelHandlerContext ctx, Object msg, ChannelPromise promise) {
				ctx.write(FederatedChunkEncoder.chunkedInput((Serializable) msg, CHUNK_SIZE, ctx.alloc(),
					ctx.pipeline().get(ChunkedWriteHandler.class)), promise);
			}
		};
	}

	private static void pumpOutbound(EmbeddedChannel channel, ChannelFuture done, List<ByteBuf> out) throws Exception {
		for(int i = 0; i < 200; i++) {
			channel.runPendingTasks();
			drainOutbound(channel, out);
			if(done.isDone())
				break;
			Thread.sleep(8);
		}
		drainOutbound(channel, out);
	}

	private static void drainOutbound(EmbeddedChannel channel, List<ByteBuf> out) {
		ByteBuf buf;
		while((buf = channel.readOutbound()) != null)
			out.add(buf);
	}

	private static FederatedResponse decode(List<ByteBuf> frames, boolean compress) throws Exception {
		EmbeddedChannel channel = compress ? new EmbeddedChannel(new JdkZlibDecoder(ZlibWrapper.ZLIB), frameDecoder(),
			new FederatedChunkDecoder()) : new EmbeddedChannel(frameDecoder(), new FederatedChunkDecoder());
		for(ByteBuf frame : frames)
			channel.writeInbound(frame);
		return awaitResponse(channel);
	}

	private static LengthFieldBasedFrameDecoder frameDecoder() {
		return new LengthFieldBasedFrameDecoder(MAX_FRAME, 1, 4, 0, 0);
	}

	private static FederatedResponse awaitResponse(EmbeddedChannel channel) throws InterruptedException {
		for(int i = 0; i < 20; i++) {
			channel.runPendingTasks();
			FederatedResponse response = channel.readInbound();
			if(response != null)
				return response;
			Thread.sleep(50);
		}
		throw new AssertionError("no decoded response");
	}

	private static void assertSamePayload(FederatedResponse expected, FederatedResponse actual) throws Exception {
		Assert.assertNotNull(actual);
		Assert.assertTrue(actual.isSuccessful());
		Assert.assertArrayEquals((double[]) expected.getData()[0], (double[]) actual.getData()[0], 0.0);
	}

	private static class Unserializable implements Serializable {
		private static final long serialVersionUID = 1L;
		private final Object _payload = new Object();

		@Override
		public String toString() {
			return String.valueOf(_payload);
		}
	}

	private static final class RecordingAllocator extends AbstractByteBufAllocator {
		private final List<ByteBuf> _frames = Collections.synchronizedList(new ArrayList<>());

		@Override
		protected ByteBuf newHeapBuffer(int initialCapacity, int maxCapacity) {
			return record(UnpooledByteBufAllocator.DEFAULT.heapBuffer(initialCapacity, maxCapacity));
		}

		@Override
		protected ByteBuf newDirectBuffer(int initialCapacity, int maxCapacity) {
			return record(UnpooledByteBufAllocator.DEFAULT.directBuffer(initialCapacity, maxCapacity));
		}

		@Override
		public boolean isDirectBufferPooled() {
			return false;
		}

		private ByteBuf record(ByteBuf buf) {
			_frames.add(buf);
			return buf;
		}

		List<ByteBuf> frames() {
			return _frames;
		}

		boolean sawFinalFrame() {
			synchronized(_frames) {
				for(ByteBuf frame : _frames)
					if(frame.isReadable() && frame.getByte(0) != FederatedChunkProtocol.TYPE_DATA)
						return true;
			}
			return false;
		}
	}
}
