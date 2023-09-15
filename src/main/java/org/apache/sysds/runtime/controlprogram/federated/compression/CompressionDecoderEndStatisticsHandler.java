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

package org.apache.sysds.runtime.controlprogram.federated.compression;

import io.netty.buffer.ByteBuf;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.ChannelInboundHandlerAdapter;
import io.netty.util.AttributeKey;
import org.apache.sysds.utils.stats.FederatedCompressionStatistics;

public class CompressionDecoderEndStatisticsHandler extends ChannelInboundHandlerAdapter {

    @Override
    public void channelRead(ChannelHandlerContext ctx, Object msg) throws Exception {
        long startTime = ctx.channel().attr(AttributeKey.<Long>valueOf("compressionDecoderStartTime")).get();
        long elapsedTime = System.currentTimeMillis() - startTime;

        ByteBuf byteBuf = (ByteBuf) msg;
        long finalSize = byteBuf.readableBytes();
        long initialSize = ctx.channel().attr(AttributeKey.<Integer>valueOf("initialSize")).get();

        FederatedCompressionStatistics.decodingStep(elapsedTime, initialSize, finalSize);
        super.channelRead(ctx, msg);
    }
}
