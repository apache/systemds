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

import io.netty.bootstrap.ServerBootstrap;
import io.netty.channel.ChannelFuture;
import io.netty.channel.ChannelInitializer;
import io.netty.channel.ChannelOption;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.nio.NioEventLoopGroup;
import io.netty.channel.socket.SocketChannel;
import io.netty.channel.socket.nio.NioServerSocketChannel;
import io.netty.handler.codec.serialization.ClassResolvers;
import io.netty.handler.codec.serialization.ObjectDecoder;
import io.netty.handler.codec.serialization.ObjectEncoder;
import org.apache.log4j.Logger;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.runtime.controlprogram.parfor.util.IDSequence;
import org.apache.sysds.runtime.instructions.cp.Data;

import java.util.HashMap;
import java.util.Map;

public class FederatedWorker {
	protected static Logger log = Logger.getLogger(FederatedWorker.class);

	private int _port;
	private int _nrThreads = Integer.parseInt(DMLConfig.DEFAULT_NUMBER_OF_FEDERATED_WORKER_THREADS);
	private IDSequence _seq = new IDSequence();
	private Map<Long, Data> _vars = new HashMap<>();

	public FederatedWorker(int port) {
		_port = (port == -1) ?
			Integer.parseInt(DMLConfig.DEFAULT_FEDERATED_PORT) : port;
	}

	public void run() {
		log.info("Setting up Federated Worker");
		EventLoopGroup bossGroup = new NioEventLoopGroup(_nrThreads);
		EventLoopGroup workerGroup = new NioEventLoopGroup(_nrThreads);
		ServerBootstrap b = new ServerBootstrap();
		b.group(bossGroup, workerGroup).channel(NioServerSocketChannel.class)
			.childHandler(new ChannelInitializer<SocketChannel>() {
				@Override
				public void initChannel(SocketChannel ch) {
					ch.pipeline()
						.addLast("ObjectDecoder",
							new ObjectDecoder(Integer.MAX_VALUE,
								ClassResolvers.weakCachingResolver(ClassLoader.getSystemClassLoader())))
						.addLast("ObjectEncoder", new ObjectEncoder())
						.addLast("FederatedWorkerHandler", new FederatedWorkerHandler(_seq, _vars));
				}
			}).option(ChannelOption.SO_BACKLOG, 128).childOption(ChannelOption.SO_KEEPALIVE, true);
		try {
			log.info("Starting Federated Worker server at port: " + _port);
			ChannelFuture f = b.bind(_port).sync();
			log.info("Started Federated Worker at port: " + _port);
			f.channel().closeFuture().sync();
		}
		catch (InterruptedException e) {
			log.error("Federated worker interrupted");
		}
		finally {
			log.info("Federated Worker Shutting down.");
			workerGroup.shutdownGracefully();
			bossGroup.shutdownGracefully();
		}
	}
}
