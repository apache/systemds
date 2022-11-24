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

import java.io.Serializable;
import java.security.cert.CertificateException;
import java.util.concurrent.SynchronousQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

import javax.net.ssl.SSLException;

import org.apache.log4j.Logger;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.CacheBlock;
import org.apache.sysds.runtime.controlprogram.paramserv.NetworkTrafficCounter;
import org.apache.sysds.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysds.runtime.lineage.LineageCache;
import org.apache.sysds.runtime.lineage.LineageCacheConfig;
import org.apache.sysds.runtime.lineage.LineageCacheConfig.ReuseCacheType;
import org.apache.sysds.runtime.lineage.LineageItem;

import io.netty.bootstrap.ServerBootstrap;
import io.netty.buffer.ByteBuf;
import io.netty.channel.ChannelFuture;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.ChannelInitializer;
import io.netty.channel.ChannelOption;
import io.netty.channel.ChannelPipeline;
import io.netty.channel.nio.NioEventLoopGroup;
import io.netty.channel.socket.SocketChannel;
import io.netty.channel.socket.nio.NioServerSocketChannel;
import io.netty.handler.codec.serialization.ObjectEncoder;
import io.netty.handler.ssl.SslContext;
import io.netty.handler.ssl.SslContextBuilder;
import io.netty.handler.ssl.util.SelfSignedCertificate;
import org.apache.sysds.runtime.controlprogram.parfor.stat.Timing;
import io.netty.handler.codec.serialization.ObjectDecoder;
import io.netty.handler.codec.serialization.ClassResolvers;

public class FederatedWorker {
	protected static Logger log = Logger.getLogger(FederatedWorker.class);

	private final int _port;
	private final FederatedLookupTable _flt;
	private final FederatedReadCache _frc;
	private final FederatedWorkloadAnalyzer _fan;
	private final boolean _debug;
	private Timing networkTimer = new Timing();

	public FederatedWorker(int port, boolean debug) {
		_flt = new FederatedLookupTable();
		_frc = new FederatedReadCache();
		if(ConfigurationManager.getCompressConfig().isWorkload())
			_fan = new FederatedWorkloadAnalyzer();
		else
			_fan = null;

		_port = (port == -1) ? DMLConfig.DEFAULT_FEDERATED_PORT : port;
		_debug = debug;

		LineageCacheConfig.setConfig(DMLScript.LINEAGE_REUSE);
		LineageCacheConfig.setCachePolicy(DMLScript.LINEAGE_POLICY);
		LineageCacheConfig.setEstimator(DMLScript.LINEAGE_ESTIMATE);

		run();
	}

	private void run() {
		log.info("Setting up Federated Worker on port " + _port);
		int par_conn = ConfigurationManager.getDMLConfig().getIntValue(DMLConfig.FEDERATED_PAR_CONN);
		final int EVENT_LOOP_THREADS = (par_conn > 0) ? par_conn : InfrastructureAnalyzer.getLocalParallelism();
		NioEventLoopGroup bossGroup = new NioEventLoopGroup(1);
		ThreadPoolExecutor workerTPE = new ThreadPoolExecutor(1, Integer.MAX_VALUE, 10, TimeUnit.SECONDS,
			new SynchronousQueue<Runnable>(true));
		NioEventLoopGroup workerGroup = new NioEventLoopGroup(EVENT_LOOP_THREADS, workerTPE);

		final boolean ssl = ConfigurationManager.isFederatedSSL();
		try {
			final ServerBootstrap b = new ServerBootstrap();
			b.group(bossGroup, workerGroup);
			b.channel(NioServerSocketChannel.class);
			b.childHandler(createChannel(ssl));
			b.option(ChannelOption.SO_BACKLOG, 128);
			b.childOption(ChannelOption.SO_KEEPALIVE, true);

			log.info("Starting Federated Worker server at port: " + _port);
			ChannelFuture f = b.bind(_port).sync();
			log.info("Started Federated Worker at port: " + _port);
			f.channel().closeFuture().sync();
		}
		catch(Exception e) {
			log.info("Federated worker interrupted");
			if(_debug) {
				log.error(e.getMessage());
				e.printStackTrace();
			}
		}
		finally {
			log.info("Federated Worker Shutting down.");
			workerGroup.shutdownGracefully();
			bossGroup.shutdownGracefully();
		}
	}

	public static class FederatedResponseEncoder extends ObjectEncoder {
		@Override
		protected ByteBuf allocateBuffer(ChannelHandlerContext ctx, Serializable msg, boolean preferDirect)
			throws Exception {
			int initCapacity = 256; // default initial capacity
			if(msg instanceof FederatedResponse) {
				FederatedResponse response = (FederatedResponse) msg;
				try {
					initCapacity = Math.toIntExact(response.estimateSerializationBufferSize());
				}
				catch(ArithmeticException ae) { // size of cache block exceeds integer limits
					initCapacity = Integer.MAX_VALUE;
				}
			}
			if(preferDirect)
				return ctx.alloc().ioBuffer(initCapacity);
			else
				return ctx.alloc().heapBuffer(initCapacity);
		}

		@Override
		protected void encode(ChannelHandlerContext ctx, Serializable msg, ByteBuf out) throws Exception {
			LineageItem objLI = null;
			boolean linReusePossible = (!ReuseCacheType.isNone() && msg instanceof FederatedResponse);
			if(linReusePossible) {
				FederatedResponse response = (FederatedResponse)msg;
				if(response.getData() != null && response.getData().length != 0
					&& response.getData()[0] instanceof CacheBlock<?>) {
					objLI = response.getLineageItem();

					byte[] cachedBytes = LineageCache.reuseSerialization(objLI);
					if(cachedBytes != null) {
						out.writeBytes(cachedBytes);
						return;
					}
				}
			}

			linReusePossible &= (objLI != null);

			int startIdx = linReusePossible ? out.writerIndex() : 0;
			long t0 = linReusePossible ? System.nanoTime() : 0;
			super.encode(ctx, msg, out);
			long t1 = linReusePossible ? System.nanoTime() : 0;

			if(linReusePossible) {
				out.readerIndex(startIdx);
				byte[] dst = new byte[out.readableBytes()];
				out.readBytes(dst);
				LineageCache.putSerializedObject(dst, objLI, (t1 - t0));
				out.resetReaderIndex();
			}
		}
	}

	private ChannelInitializer<SocketChannel> createChannel(boolean ssl) {
		try {
			// TODO add ability to use real ssl files, not self signed certificates.
			final SelfSignedCertificate cert = new SelfSignedCertificate();
			final SslContext cont2 = SslContextBuilder.forServer(cert.certificate(), cert.privateKey()).build();

			return new ChannelInitializer<SocketChannel>() {
				@Override
				public void initChannel(SocketChannel ch) {
					final ChannelPipeline cp = ch.pipeline();
					if(ConfigurationManager.getDMLConfig()
						.getBooleanValue(DMLConfig.USE_SSL_FEDERATED_COMMUNICATION)) {
						cp.addLast(cont2.newHandler(ch.alloc()));
					}
					if(ssl)
						cp.addLast(cont2.newHandler(ch.alloc()));
					cp.addLast("NetworkTrafficCounter", new NetworkTrafficCounter(FederatedStatistics::logWorkerTraffic));
					cp.addLast("ObjectDecoder",
						new ObjectDecoder(Integer.MAX_VALUE,
							ClassResolvers.weakCachingResolver(ClassLoader.getSystemClassLoader())));
					cp.addLast("ObjectEncoder", new ObjectEncoder());
					cp.addLast(FederationUtils.decoder(), new FederatedResponseEncoder());
					cp.addLast(new FederatedWorkerHandler(_flt, _frc, _fan, networkTimer));
				}
			};
		}
		catch(CertificateException | SSLException e) {
			throw new DMLRuntimeException("Failed creating channel SSL", e);
		}
	}
}
