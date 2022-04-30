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
import io.netty.handler.codec.serialization.ClassResolvers;
import io.netty.handler.codec.serialization.ObjectDecoder;
import io.netty.handler.codec.serialization.ObjectEncoder;
import io.netty.handler.ssl.SslContext;
import io.netty.handler.ssl.SslContextBuilder;
import io.netty.handler.ssl.util.SelfSignedCertificate;
import org.apache.sysds.api.DMLScript;
import org.apache.log4j.Logger;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysds.runtime.lineage.LineageCacheConfig;

public class FederatedWorker {
	protected static Logger log = Logger.getLogger(FederatedWorker.class);

	private final int _port;
	private final FederatedLookupTable _flt;
	private final FederatedReadCache _frc;
	private final FederatedWorkloadAnalyzer _fan;
	private final boolean _debug;

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
	}

	public void run() throws CertificateException, SSLException {
		log.info("Setting up Federated Worker on port " + _port);
		int par_conn = ConfigurationManager.getDMLConfig().getIntValue(DMLConfig.FEDERATED_PAR_CONN);
		final int EVENT_LOOP_THREADS = (par_conn > 0) ? par_conn : InfrastructureAnalyzer.getLocalParallelism();
		NioEventLoopGroup bossGroup = new NioEventLoopGroup(1);
		ThreadPoolExecutor workerTPE = new ThreadPoolExecutor(1, Integer.MAX_VALUE,
			10, TimeUnit.SECONDS, new SynchronousQueue<Runnable>(true));
		NioEventLoopGroup workerGroup = new NioEventLoopGroup(EVENT_LOOP_THREADS, workerTPE);
		ServerBootstrap b = new ServerBootstrap();
		// TODO add ability to use real ssl files, not self signed certificates.
		SelfSignedCertificate cert = new SelfSignedCertificate();
		final SslContext cont2 = SslContextBuilder.forServer(cert.certificate(), cert.privateKey()).build();

		try {
			b.group(bossGroup, workerGroup).channel(NioServerSocketChannel.class)
				.childHandler(new ChannelInitializer<SocketChannel>() {
					@Override
					public void initChannel(SocketChannel ch) {
						ChannelPipeline cp = ch.pipeline();

						if(ConfigurationManager.getDMLConfig()
							.getBooleanValue(DMLConfig.USE_SSL_FEDERATED_COMMUNICATION)) {
							cp.addLast(cont2.newHandler(ch.alloc()));
						}
						cp.addLast("ObjectDecoder",
							new ObjectDecoder(Integer.MAX_VALUE,
								ClassResolvers.weakCachingResolver(ClassLoader.getSystemClassLoader())));
						cp.addLast("FederatedResponseEncoder", new FederatedResponseEncoder());
						cp.addLast("FederatedWorkerHandler", new FederatedWorkerHandler(_flt, _frc, _fan));
					}
				}).option(ChannelOption.SO_BACKLOG, 128).childOption(ChannelOption.SO_KEEPALIVE, true);
			log.info("Starting Federated Worker server at port: " + _port);
			ChannelFuture f = b.bind(_port).sync();
			log.info("Started Federated Worker at port: " + _port);
			f.channel().closeFuture().sync();
		}
		catch(Exception e) {
			log.info("Federated worker interrupted");
			if ( _debug ){
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
		protected ByteBuf allocateBuffer(ChannelHandlerContext ctx, Serializable msg,
			boolean preferDirect) throws Exception {
			int initCapacity = 256; // default initial capacity
			if(msg instanceof FederatedResponse) {
				FederatedResponse response = (FederatedResponse)msg;
				try {
					initCapacity = Math.toIntExact(response.estimateSerializationBufferSize());
				} catch(ArithmeticException ae) { // size of cache block exceeds integer limits
					initCapacity = Integer.MAX_VALUE;
				}
			}
			if(preferDirect)
				return ctx.alloc().ioBuffer(initCapacity);
			else
				return ctx.alloc().heapBuffer(initCapacity);
		}
	}
}
