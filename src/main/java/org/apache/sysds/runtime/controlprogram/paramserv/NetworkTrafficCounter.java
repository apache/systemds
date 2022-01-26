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

package org.apache.sysds.runtime.controlprogram.paramserv;

import io.netty.channel.ChannelHandler;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.traffic.ChannelTrafficShapingHandler;
import java.util.function.BiConsumer;

public class NetworkTrafficCounter extends ChannelTrafficShapingHandler {
    private final BiConsumer<Long, Long> _fn; // (read, written) -> Void, logs bytes read and written
    public NetworkTrafficCounter(BiConsumer<Long, Long> fn) {
        // checkInterval of zero means that doAccounting will not be called
        super( 0);
        _fn = fn;
    }

    // log bytes read/written after channel is closed
    @Override
    public void channelInactive(ChannelHandlerContext ctx) throws Exception {
        _fn.accept(trafficCounter.cumulativeReadBytes(), trafficCounter.cumulativeWrittenBytes());
        trafficCounter.resetCumulativeTime();
        super.channelInactive(ctx);
    }
}
