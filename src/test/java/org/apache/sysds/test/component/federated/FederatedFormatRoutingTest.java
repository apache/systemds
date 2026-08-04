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

import java.util.ArrayList;
import java.util.List;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.runtime.controlprogram.federated.FederatedFormatDecoder;
import org.apache.sysds.runtime.controlprogram.federated.FederatedFormatEncoder;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse.ResponseType;
import org.apache.sysds.runtime.lineage.LineageCacheConfig.ReuseCacheType;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.junit.Assert;
import org.junit.Test;

import io.netty.buffer.ByteBuf;
import io.netty.channel.ChannelFuture;
import io.netty.channel.embedded.EmbeddedChannel;
import io.netty.handler.codec.serialization.ObjectEncoder;
import io.netty.handler.stream.ChunkedWriteHandler;

public class FederatedFormatRoutingTest {
	private static final int CHUNK_SIZE = 4096; // tiny on purpose: forces a multi-frame chunk stream
	private static final long THRESHOLD_NEVER = Long.MAX_VALUE; // size guard never trips
	private static final long THRESHOLD_ALWAYS = 1; // size guard always trips -> stream
	private static final int PAYLOAD_DOUBLES = 20000; // ~160 KB serialized
	private static final byte MARKER_OBJECT_ENCODER = 0;
	private static final byte MARKER_CHUNKED = 1;

	@Test
	public void nonCacheableRoutesChunked() throws Exception {
		// plain double[] payload is not lineage-cacheable -> streams regardless of threshold
		FederatedResponse original = sampleResponse();
		List<ByteBuf> wire = encode(original, THRESHOLD_NEVER);
		Assert.assertEquals(MARKER_CHUNKED, marker(wire));

		EmbeddedChannel in = new EmbeddedChannel(new FederatedFormatDecoder());
		FederatedResponse decoded = decode(in, wire);
		Assert.assertNotNull(in.pipeline().get("FederatedChunkDecoder"));
		assertDetectorRemoved(in);
		assertSamePayload(original, decoded);
	}

	@Test
	public void lineageCacheableRoutesObjectEncoder() throws Exception {
		ReuseCacheType prev = DMLScript.LINEAGE_REUSE;
		DMLScript.LINEAGE_REUSE = ReuseCacheType.REUSE_FULL;
		try {
			List<ByteBuf> wire = encode(cacheableResponse(), THRESHOLD_NEVER);
			Assert.assertEquals(MARKER_OBJECT_ENCODER, marker(wire));

			EmbeddedChannel in = new EmbeddedChannel(new FederatedFormatDecoder());
			FederatedResponse decoded = decode(in, wire);
			Assert.assertNotNull(in.pipeline().get("FederatedObjectDecoder"));
			assertDetectorRemoved(in);
			Assert.assertTrue(decoded.isSuccessful());
		}
		finally {
			DMLScript.LINEAGE_REUSE = prev;
		}
	}

	@Test
	public void lineageCacheableOverThresholdRoutesChunked() throws Exception {
		ReuseCacheType prev = DMLScript.LINEAGE_REUSE;
		DMLScript.LINEAGE_REUSE = ReuseCacheType.REUSE_FULL;
		try {
			List<ByteBuf> wire = encode(cacheableResponse(), THRESHOLD_ALWAYS);
			Assert.assertEquals(MARKER_CHUNKED, marker(wire));
		}
		finally {
			DMLScript.LINEAGE_REUSE = prev;
		}
	}

	@Test
	public void reuseDisabledCacheableRoutesChunked() throws Exception {
		// cacheable payload but lineage reuse off -> no cache to feed -> streams
		ReuseCacheType prev = DMLScript.LINEAGE_REUSE;
		DMLScript.LINEAGE_REUSE = ReuseCacheType.NONE;
		try {
			List<ByteBuf> wire = encode(cacheableResponse(), THRESHOLD_NEVER);
			Assert.assertEquals(MARKER_CHUNKED, marker(wire));
		}
		finally {
			DMLScript.LINEAGE_REUSE = prev;
		}
	}

	private static FederatedResponse sampleResponse() {
		double[] data = new double[PAYLOAD_DOUBLES];
		for(int i = 0; i < data.length; i++)
			data[i] = i;
		return new FederatedResponse(ResponseType.SUCCESS, data);
	}

	private static FederatedResponse cacheableResponse() {
		MatrixBlock mb = new MatrixBlock(16, 16, 1.0);
		return new FederatedResponse(ResponseType.SUCCESS, new Object[] {mb}, new LineageItem("routing-test"));
	}

	private static List<ByteBuf> encode(FederatedResponse response, long threshold) throws Exception {
		EmbeddedChannel out = new EmbeddedChannel(new ChunkedWriteHandler(), new ObjectEncoder(),
			new FederatedFormatEncoder(CHUNK_SIZE, threshold));
		out.config().setWriteBufferHighWaterMark((CHUNK_SIZE + 64) * 64);
		List<ByteBuf> wire = new ArrayList<>();
		ChannelFuture done = out.write(response);
		out.flush();
		for(int i = 0; i < 800; i++) {
			out.runPendingTasks();
			drainInto(out, wire);
			if(done.isDone())
				break;
			Thread.sleep(2);
		}
		drainInto(out, wire);
		return wire;
	}

	private static void drainInto(EmbeddedChannel out, List<ByteBuf> wire) {
		ByteBuf buf;
		while((buf = out.readOutbound()) != null)
			wire.add(buf);
	}

	private static byte marker(List<ByteBuf> wire) {
		ByteBuf first = wire.get(0);
		Assert.assertEquals("marker is a standalone 1-byte frame", 1, first.readableBytes());
		return first.getByte(first.readerIndex());
	}

	private static FederatedResponse decode(EmbeddedChannel in, List<ByteBuf> wire) throws Exception {
		for(ByteBuf buf : wire)
			in.writeInbound(buf);
		for(int i = 0; i < 200; i++) {
			in.runPendingTasks();
			FederatedResponse response = in.readInbound();
			if(response != null)
				return response;
			Thread.sleep(5);
		}
		throw new AssertionError("no decoded response");
	}

	private static void assertDetectorRemoved(EmbeddedChannel in) {
		Assert.assertNull("detector must remove itself after the first message",
			in.pipeline().get(FederatedFormatDecoder.class));
	}

	private static void assertSamePayload(FederatedResponse expected, FederatedResponse actual) throws Exception {
		Assert.assertNotNull(actual);
		Assert.assertTrue(actual.isSuccessful());
		Assert.assertArrayEquals((double[]) expected.getData()[0], (double[]) actual.getData()[0], 0.0);
	}
}
