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

import java.util.List;

import io.netty.buffer.ByteBuf;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.ChannelPipeline;
import io.netty.handler.codec.ByteToMessageDecoder;

public final class FederatedFormatDecoder extends ByteToMessageDecoder {
	@Override
	protected void decode(ChannelHandlerContext ctx, ByteBuf in, List<Object> out) {
		if(in.readableBytes() < 1)
			return;
		byte marker = in.readByte();
		ChannelPipeline cp = ctx.pipeline();
		if(marker == FederatedChunkProtocol.MARKER_CHUNKED) {
			cp.addAfter(ctx.name(), "FederatedFrameDecoder", FederatedChunkProtocol.newFrameDecoder());
			cp.addAfter("FederatedFrameDecoder", "FederatedChunkDecoder", new FederatedChunkDecoder());
		}
		else {
			cp.addAfter(ctx.name(), "FederatedObjectDecoder", FederationUtils.decoder());
		}
		cp.remove(this);
	}
}
