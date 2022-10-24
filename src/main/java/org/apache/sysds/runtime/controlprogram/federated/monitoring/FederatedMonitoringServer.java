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

package org.apache.sysds.runtime.controlprogram.federated.monitoring;

import org.apache.log4j.Logger;

import io.netty.bootstrap.ServerBootstrap;
import io.netty.channel.Channel;
import io.netty.channel.ChannelFuture;
import io.netty.channel.ChannelInitializer;
import io.netty.channel.ChannelOption;
import io.netty.channel.ChannelPipeline;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.nio.NioEventLoopGroup;
import io.netty.channel.socket.nio.NioServerSocketChannel;
import io.netty.handler.codec.http.HttpMethod;
import io.netty.handler.codec.http.HttpServerCodec;
import io.netty.handler.codec.http.cors.CorsConfigBuilder;
import io.netty.handler.codec.http.cors.CorsHandler;

public class FederatedMonitoringServer {
	protected static Logger log = Logger.getLogger(FederatedMonitoringServer.class);
	private final int _port;

	private final boolean _debug;

	public FederatedMonitoringServer(int port, boolean debug) {
		_port = (port == -1) ? 4201 : port;

		_debug = debug;

		run();
	}

	public void run() {
		log.info("Setting up Federated Monitoring Backend on port " + _port);
		EventLoopGroup bossGroup = new NioEventLoopGroup();
		EventLoopGroup workerGroup = new NioEventLoopGroup();

		try {
			var corsConfig = CorsConfigBuilder.forAnyOrigin()
					.allowedRequestHeaders("*")
					.allowedRequestMethods(
							HttpMethod.DELETE,
							HttpMethod.GET,
							HttpMethod.PUT,
							HttpMethod.POST,
							HttpMethod.OPTIONS)
					.build();

			ServerBootstrap server = new ServerBootstrap();
			server.group(bossGroup, workerGroup)
				.channel(NioServerSocketChannel.class)
				.childHandler(new ChannelInitializer<>() {
					@Override
					protected void initChannel(Channel ch) {
					ChannelPipeline pipeline = ch.pipeline();

					pipeline.addLast(new HttpServerCodec());
					pipeline.addLast(new CorsHandler(corsConfig));
					pipeline.addLast(new FederatedMonitoringServerHandler());
					}
				});

			server.childOption(ChannelOption.SO_KEEPALIVE, true);

			log.info("Starting Federated Monitoring Backend server at port: " + _port);
			ChannelFuture f = server.bind(_port).sync();
			log.info("Started Federated Monitoring Backend at port: " + _port);
			f.channel().closeFuture().sync();
		} catch(Exception e) {
			log.info("Federated Monitoring Backend Interrupted");
			if (_debug) {
				log.error(e.getMessage());
				e.printStackTrace();
			}
		} finally{
			log.info("Federated Monitoring Backend Shutting down.");
			workerGroup.shutdownGracefully();
			bossGroup.shutdownGracefully();
		}
	}
}
