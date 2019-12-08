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
import org.tugraz.sysds.runtime.controlprogram.parfor.util.IDSequence;
import org.tugraz.sysds.runtime.instructions.cp.Data;

import java.util.HashMap;
import java.util.Map;

public class FederatedWorker {
	
	public int _port;
	private IDSequence _seq = new IDSequence();
	private Map<Long, Data> _vars = new HashMap<>();
	
	public FederatedWorker(int port) {
		_port = port;
	}
	
	public void run() throws Exception {
		//TODO modify logging to use actual logger not stdout,
		//otherwise the stdout queue might fill up if nobody is reading from it
		System.out.println("[Federated Worker] Setting up...");
		EventLoopGroup bossGroup = new NioEventLoopGroup();
		EventLoopGroup workerGroup = new NioEventLoopGroup();
		try {
			ServerBootstrap b = new ServerBootstrap();
			b.group(bossGroup, workerGroup)
				.channel(NioServerSocketChannel.class)
				.childHandler(new ChannelInitializer<SocketChannel>() {
					@Override
					public void initChannel(SocketChannel ch) {
						ch.pipeline().addLast("ObjectDecoder",
							new ObjectDecoder(ClassResolvers.weakCachingResolver(ClassLoader.getSystemClassLoader())))
								.addLast("ObjectEncoder", new ObjectEncoder())
								.addLast("FederatedWorkerHandler", new FederatedWorkerHandler(_seq, _vars));
					}
				})
				.option(ChannelOption.SO_BACKLOG, 128)
				.childOption(ChannelOption.SO_KEEPALIVE, true);
			System.out.println("[Federated Worker] Starting server at port " + _port);
			ChannelFuture f = b.bind(_port).sync();
			f.channel().closeFuture().sync();
		}
		finally {
			System.out.println("[Federated Worker] Shutting down.");
			workerGroup.shutdownGracefully();
			bossGroup.shutdownGracefully();
		}
	}
}
