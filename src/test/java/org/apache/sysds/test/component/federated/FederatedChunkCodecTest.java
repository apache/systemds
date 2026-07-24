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

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import org.apache.sysds.runtime.controlprogram.federated.FederatedChunkDecoder;
import org.apache.sysds.runtime.controlprogram.federated.FederatedChunkEncoder;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse.ResponseType;
import org.junit.Assert;
import org.junit.Test;

import io.netty.buffer.ByteBuf;
import io.netty.channel.ChannelFuture;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.ChannelOutboundHandlerAdapter;
import io.netty.channel.ChannelPromise;
import io.netty.channel.embedded.EmbeddedChannel;
import io.netty.handler.codec.LengthFieldBasedFrameDecoder;
import io.netty.handler.codec.compression.JdkZlibDecoder;
import io.netty.handler.codec.compression.JdkZlibEncoder;
import io.netty.handler.codec.compression.ZlibWrapper;
import io.netty.handler.stream.ChunkedWriteHandler;

public class FederatedChunkCodecTest {
	private static final int CHUNK_SIZE = 4096; // tiny on purpose: forces a multi-frame stream
	private static final int MAX_FRAME = 1 << 20; // 1 MB: frame-decoder ceiling, must exceed CHUNK_SIZE + header
	private static final int PAYLOAD_DOUBLES = 20000; // ~160 KB serialized, many CHUNK_SIZE frames

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

	private static FederatedResponse sampleResponse() {
		double[] data = new double[PAYLOAD_DOUBLES];
		for(int i = 0; i < data.length; i++)
			data[i] = i;
		return new FederatedResponse(ResponseType.SUCCESS, data);
	}

	private static List<ByteBuf> encode(FederatedResponse response, boolean compress) throws Exception {
		EmbeddedChannel channel = compress
			? new EmbeddedChannel(new JdkZlibEncoder(ZlibWrapper.ZLIB), new ChunkedWriteHandler(), chunkEncoder())
			: new EmbeddedChannel(new ChunkedWriteHandler(), chunkEncoder());
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
		for(int i = 0; i < 800; i++) {
			channel.runPendingTasks();
			drainOutbound(channel, out);
			if(done.isDone())
				break;
			Thread.sleep(2);
		}
		drainOutbound(channel, out);
	}

	private static void drainOutbound(EmbeddedChannel channel, List<ByteBuf> out) {
		ByteBuf buf;
		while((buf = channel.readOutbound()) != null)
			out.add(buf);
	}

	private static FederatedResponse decode(List<ByteBuf> frames, boolean compress) throws Exception {
		EmbeddedChannel channel = compress
			? new EmbeddedChannel(new JdkZlibDecoder(ZlibWrapper.ZLIB), frameDecoder(), new FederatedChunkDecoder())
			: new EmbeddedChannel(frameDecoder(), new FederatedChunkDecoder());
		for(ByteBuf frame : frames)
			channel.writeInbound(frame);
		return awaitResponse(channel);
	}

	private static LengthFieldBasedFrameDecoder frameDecoder() {
		return new LengthFieldBasedFrameDecoder(MAX_FRAME, 1, 4, 0, 0);
	}

	private static FederatedResponse awaitResponse(EmbeddedChannel channel) throws InterruptedException {
		for(int i = 0; i < 200; i++) {
			channel.runPendingTasks();
			FederatedResponse response = channel.readInbound();
			if(response != null)
				return response;
			Thread.sleep(5);
		}
		throw new AssertionError("no decoded response");
	}

	private static void assertSamePayload(FederatedResponse expected, FederatedResponse actual) throws Exception {
		Assert.assertNotNull(actual);
		Assert.assertTrue(actual.isSuccessful());
		Assert.assertArrayEquals((double[]) expected.getData()[0], (double[]) actual.getData()[0], 0.0);
	}
}
