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
