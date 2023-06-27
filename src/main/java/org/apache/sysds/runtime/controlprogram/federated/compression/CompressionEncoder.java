package org.apache.sysds.runtime.controlprogram.federated.compression;

import io.netty.buffer.ByteBuf;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.compression.JdkZlibEncoder;
import org.apache.log4j.Logger;

public class CompressionEncoder extends JdkZlibEncoder {
    protected static Logger log = Logger.getLogger(CompressionEncoder.class);

    @Override
    protected void encode(ChannelHandlerContext ctx, ByteBuf in, ByteBuf out) throws Exception {
        int originalSize = in.readableBytes();
        super.encode(ctx, in, out);
        int compressedSize = out.readableBytes();

        // Calculate and log the compression ratio
        double compressionRatio = 100.0 * (1.0 - (double) compressedSize / originalSize);
        log.debug("Original message length: " + originalSize);
        log.debug("Compressed size: " + compressedSize);
        log.debug("Compression ratio: " + compressionRatio + "%");
    }
}
