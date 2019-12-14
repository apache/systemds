/*
 * Copyright 2019 Graz University of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package org.tugraz.sysds.runtime.controlprogram.federated;


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
import org.tugraz.sysds.runtime.controlprogram.caching.CacheableData;
import org.tugraz.sysds.runtime.controlprogram.parfor.util.IDSequence;

import java.util.HashMap;
import java.util.Map;

public class FederatedWorker {
	protected static Logger log = Logger.getLogger(FederatedWorker.class);
	
	public int _port;
	private IDSequence _seq = new IDSequence();
	private Map<Long, CacheableData<?>> _vars = new HashMap<>();
	
	public FederatedWorker(int port) {
		_port = port;
	}
	
	public void run() throws Exception {
		log.debug("[Federated Worker] Setting up...");
		EventLoopGroup bossGroup = new NioEventLoopGroup(10);
		EventLoopGroup workerGroup = new NioEventLoopGroup(10);
		try {
			ServerBootstrap b = new ServerBootstrap();
			b.group(bossGroup, workerGroup)
				.channel(NioServerSocketChannel.class)
				.childHandler(new ChannelInitializer<SocketChannel>() {
					@Override
					public void initChannel(SocketChannel ch) {
						ch.pipeline().addLast("ObjectDecoder",
							new ObjectDecoder(Integer.MAX_VALUE, ClassResolvers.weakCachingResolver(ClassLoader.getSystemClassLoader())))
								.addLast("ObjectEncoder", new ObjectEncoder())
								.addLast("FederatedWorkerHandler", new FederatedWorkerHandler(_seq, _vars));
					}
				})
				.option(ChannelOption.SO_BACKLOG, 128)
				.childOption(ChannelOption.SO_KEEPALIVE, true);
			log.debug("[Federated Worker] Starting server at port " + _port);
			ChannelFuture f = b.bind(_port).sync();
			f.channel().closeFuture().sync();
		}
		finally {
			log.debug("[Federated Worker] Shutting down.");
			workerGroup.shutdownGracefully();
			bossGroup.shutdownGracefully();
		}
	}
}
