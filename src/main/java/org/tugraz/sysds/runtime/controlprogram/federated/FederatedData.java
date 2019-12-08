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

import io.netty.bootstrap.Bootstrap;
import io.netty.channel.Channel;
import io.netty.channel.ChannelFuture;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.ChannelInboundHandlerAdapter;
import io.netty.channel.ChannelInitializer;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.nio.NioEventLoopGroup;
import io.netty.channel.socket.SocketChannel;
import io.netty.channel.socket.nio.NioSocketChannel;
import io.netty.handler.codec.serialization.ClassResolvers;
import io.netty.handler.codec.serialization.ObjectDecoder;
import io.netty.handler.codec.serialization.ObjectEncoder;
import org.tugraz.sysds.runtime.DMLRuntimeException;

import java.net.InetSocketAddress;
import java.util.concurrent.CompletableFuture;

public class FederatedData {
	private InetSocketAddress _address;
	private String _filepath;
	/**
	 * The ID of default matrix/tensor on which operations get executed if no other ID is given.
	 */
	private long _varID = -1; // -1 is never valid since varIDs start at 0
	
	public FederatedData(InetSocketAddress address, String filepath) {
		_address = address;
		_filepath = filepath;
	}
	
	public InetSocketAddress getAddress() {
		return _address;
	}
	
	public long getVarID() {
		return _varID;
	}
	
	public String getFilepath() {
		return _filepath;
	}
	
	public boolean isInitialized() {
		return _varID != -1;
	}
	
	public void initFederatedData() {
		if( isInitialized() )
			throw new DMLRuntimeException("Tried to init already initialized data");
		FederatedRequest request = new FederatedRequest(FederatedRequest.FedMethod.READ);
		request.appendParam(_filepath);
		FederatedResponse response = executeFederatedOperation(request);
		if( response.isSuccessful() )
			_varID = (Long) response.getData();
		else
			throw new DMLRuntimeException(response.getErrorMessage());
	}
	
	/**
	 * Executes an federated operation on a federated worker and default variable.
	 *
	 * @param request the requested operation
	 * @param withVarID true if we should add the default varID (initialized) or false if we should not
	 * @return the response
	 */
	public FederatedResponse executeFederatedOperation(FederatedRequest request, boolean withVarID) {
		if (withVarID) {
			if( !isInitialized() )
				throw new DMLRuntimeException("Tried to execute federated operation on data non initialized federated data.");
			return executeFederatedOperation(request, _varID);
		}
		return executeFederatedOperation(request);
	}
	
	/**
	 * Executes an federated operation on a federated worker.
	 *
	 * @param request the requested operation
	 * @return the response
	 */
	public FederatedResponse executeFederatedOperation(FederatedRequest request, long varID) {
		request.appendParam(varID);
		return executeFederatedOperation(request);
	}
	
	/**
	 * Executes an federated operation on a federated worker.
	 *
	 * @param request the requested operation
	 * @return the response
	 */
	public FederatedResponse executeFederatedOperation(FederatedRequest request) {
		// TODO async
		EventLoopGroup workerGroup = new NioEventLoopGroup();
		try {
			Bootstrap b = new Bootstrap();
			CompletableFuture<FederatedResponse> future = new CompletableFuture<>();
			b.group(workerGroup).channel(NioSocketChannel.class).handler(new ChannelInitializer<SocketChannel>() {
				@Override
				public void initChannel(SocketChannel ch) throws Exception {
					ch.pipeline().addLast("ObjectDecoder",
					new ObjectDecoder(ClassResolvers.weakCachingResolver(ClassLoader.getSystemClassLoader())))
					.addLast("FederatedOperationHandler", new ChannelInboundHandlerAdapter() {
						@Override
						public void channelRead(ChannelHandlerContext ctx, Object msg) throws Exception {
							future.complete((FederatedResponse) msg);
						}
					})
					.addLast("ObjectEncoder", new ObjectEncoder());
				}
			});
			
			ChannelFuture f = b.connect(_address).sync();
			Channel c = f.channel();
			c = c.writeAndFlush(request).sync().channel();
			c.read();
			FederatedResponse response = future.get();
			c.close().sync();
			return response;
		}
		catch (InterruptedException e) {
			throw new DMLRuntimeException("Could not send federated operation.");
		}
		catch (Exception e) {
			throw new DMLRuntimeException(e);
		}
		finally {
			workerGroup.shutdownGracefully();
		}
	}
}
