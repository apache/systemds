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
import java.net.InetSocketAddress;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.Future;

import javax.net.ssl.SSLException;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.CacheBlock;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest.RequestType;
import org.apache.sysds.runtime.controlprogram.paramserv.NetworkTrafficCounter;
import org.apache.sysds.runtime.meta.MetaData;

import io.netty.bootstrap.Bootstrap;
import io.netty.buffer.ByteBuf;
import io.netty.channel.ChannelFuture;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.ChannelInboundHandlerAdapter;
import io.netty.channel.ChannelInitializer;
import io.netty.channel.ChannelPipeline;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.nio.NioEventLoopGroup;
import io.netty.channel.socket.SocketChannel;
import io.netty.channel.socket.nio.NioSocketChannel;
import io.netty.handler.codec.serialization.ObjectEncoder;
import io.netty.handler.ssl.SslContext;
import io.netty.handler.ssl.SslContextBuilder;
import io.netty.handler.ssl.SslHandler;
import io.netty.handler.ssl.util.InsecureTrustManagerFactory;
import io.netty.handler.timeout.ReadTimeoutHandler;
import io.netty.util.concurrent.Promise;

public class FederatedData {
	private static final Log LOG = LogFactory.getLog(FederatedData.class.getName());
	private static final Set<InetSocketAddress> _allFedSites = new HashSet<>();

	/** Thread pool specific for the federated requests */
	private static EventLoopGroup workerGroup = null;

	/** A Singleton constructed SSL context, that only is assigned if ssl is enabled. */
	private static SslContextMan sslInstance = null;

	private final Types.DataType _dataType;
	private final InetSocketAddress _address;
	private final String _filepath;

	/**
	 * The ID of default matrix/tensor on which operations get executed if no other ID is given.
	 */
	private long _varID = -1; // -1 is never valid since varIDs start at 0

	public FederatedData(Types.DataType dataType, InetSocketAddress address, String filepath) {
		_dataType = dataType;
		_address = address;
		_filepath = filepath;
		if(_address != null)
			_allFedSites.add(_address);
	}

	public FederatedData(Types.DataType dataType, InetSocketAddress address, String filepath, long varID) {
		_dataType = dataType;
		_address = address;
		_filepath = filepath;
		_varID = varID;
	}

	public InetSocketAddress getAddress() {
		return _address;
	}

	public void setVarID(long varID) {
		_varID = varID;
	}

	public long getVarID() {
		return _varID;
	}

	public String getFilepath() {
		return _filepath;
	}

	public Types.DataType getDataType() {
		return _dataType;
	}

	public boolean isInitialized() {
		return _varID != -1;
	}

	boolean equalAddress(FederatedData that) {
		return _address != null && that != null && that._address != null && _address.equals(that._address);
	}

	/**
	 * Make a copy of the <code>FederatedData</code> metadata, but use another varID (refer to another object on worker)
	 * 
	 * @param varID the varID of the variable we refer to
	 * @return new <code>FederatedData</code> with different varID set
	 */
	public FederatedData copyWithNewID(long varID) {
		FederatedData copy = new FederatedData(_dataType, _address, _filepath);
		copy.setVarID(varID);
		return copy;
	}

	public synchronized Future<FederatedResponse> initFederatedData(long id) {
		return initFederatedData(id, null);
	}

	public synchronized Future<FederatedResponse> initFederatedData(long id, MetaData mtd) {
		if(isInitialized())
			throw new DMLRuntimeException("Tried to init already initialized data");
		if(!_dataType.isMatrix() && !_dataType.isFrame())
			throw new DMLRuntimeException("Federated datatype \"" + _dataType.toString() + "\" is not supported.");
		_varID = id;
		FederatedRequest request = (mtd != null) ?
			new FederatedRequest(RequestType.READ_VAR, id, mtd) :
			new FederatedRequest(RequestType.READ_VAR, id);
		request.appendParam(_filepath);
		request.appendParam(_dataType.name());
		return executeFederatedOperation(request);
	}

	public synchronized Future<FederatedResponse> initFederatedDataFromLocal(long id, CacheBlock<?> block) {
		if(isInitialized())
			throw new DMLRuntimeException("Tried to init already initialized data");
		if(!_dataType.isMatrix() && !_dataType.isFrame())
			throw new DMLRuntimeException("Federated datatype \"" + _dataType.toString() + "\" is not supported.");
		_varID = id;
		FederatedRequest request = new FederatedRequest(RequestType.READ_VAR, id);
		request.appendParam(_filepath);
		request.appendParam(_dataType.name());
		request.appendParam(block);
		return executeFederatedOperation(request);
	}

	public Future<FederatedResponse> executeFederatedOperation(FederatedRequest... request) {
		return executeFederatedOperation(_address, request);
	}

	/**
	 * Executes an federated operation on a federated worker.
	 *
	 * @param address socket address (incl host and port)
	 * @param request the requested operation
	 * @return the response
	 */
	public synchronized static Future<FederatedResponse> executeFederatedOperation(InetSocketAddress address,
		FederatedRequest... request) {
		try {
			final Bootstrap b = new Bootstrap();
			if(workerGroup == null)
				createWorkGroup();
			b.group(workerGroup);
			b.channel(NioSocketChannel.class);
			final DataRequestHandler handler = new DataRequestHandler();
			// Client Netty

			b.handler(createChannel(address, handler));

			ChannelFuture f = b.connect(address).sync();
			Promise<FederatedResponse> promise = f.channel().eventLoop().newPromise();
			handler.setPromise(promise);
			f.channel().writeAndFlush(request);

			return handler.getProm();
		}
		catch(Exception e) {
			throw new DMLRuntimeException("Failed sending federated operation", e);
		}
	}

	private static ChannelInitializer<SocketChannel> createChannel(InetSocketAddress address, DataRequestHandler handler){
		final int timeout = ConfigurationManager.getFederatedTimeout();
		final boolean ssl = ConfigurationManager.isFederatedSSL();

		return new ChannelInitializer<SocketChannel>() {
			@Override
			protected void initChannel(SocketChannel ch) throws Exception {
				final ChannelPipeline cp = ch.pipeline();
				cp.addLast("NetworkTrafficCounter", new NetworkTrafficCounter(FederatedStatistics::logServerTraffic));
				if(ssl)
					cp.addLast(createSSLHandler(ch, address));
				if(timeout > -1)
					cp.addLast(new ReadTimeoutHandler(timeout));
				cp.addLast(FederationUtils.decoder(), new FederatedRequestEncoder(), handler);
			}
		};
	}

	public static void clearFederatedWorkers() {
		if(_allFedSites.isEmpty())
			return;

		try {
			// create and execute clear request on all workers
			FederatedRequest fr = new FederatedRequest(RequestType.CLEAR);
			List<Future<FederatedResponse>> ret = new ArrayList<>();
			for(InetSocketAddress address : _allFedSites)
				ret.add(executeFederatedOperation(address, fr));

			// wait for successful completion
			FederationUtils.waitFor(ret);
		}
		catch(Exception ex) {
			LOG.warn("Failed to execute CLEAR request on existing federated sites.", ex);
		}
		finally {
			resetFederatedSites();
		}
	}

	private static SslHandler createSSLHandler(SocketChannel ch, InetSocketAddress address){
		return SslConstructor().context.newHandler(ch.alloc(), address.getAddress().getHostAddress(),
							address.getPort());
	}

	public static void resetFederatedSites() {
		_allFedSites.clear();
	}

	public static void clearWorkGroup() {
		if(workerGroup != null)
			workerGroup.shutdownGracefully();
		workerGroup = null;
	}

	public synchronized static void createWorkGroup() {
		if(workerGroup == null)
			workerGroup = new NioEventLoopGroup(DMLConfig.DEFAULT_NUMBER_OF_FEDERATED_WORKER_THREADS);
	}

	private static class DataRequestHandler extends ChannelInboundHandlerAdapter {
		private Promise<FederatedResponse> _prom;

		public DataRequestHandler() {
		}

		public void setPromise(Promise<FederatedResponse> prom) {
			_prom = prom;
		}

		@Override
		public void channelRead(ChannelHandlerContext ctx, Object msg) {
			_prom.setSuccess((FederatedResponse) msg);
			ctx.close();
		}

		public Promise<FederatedResponse> getProm() {
			return _prom;
		}
	}

	private static class SslContextMan {
		protected final SslContext context;

		private SslContextMan() {
			try {
				context = SslContextBuilder.forClient().trustManager(InsecureTrustManagerFactory.INSTANCE).build();
			}
			catch(SSLException e) {
				throw new DMLRuntimeException("Static SSL setup failed for client side", e);
			}
		}
	}

	private static SslContextMan SslConstructor() {
		if(sslInstance == null)
			return new SslContextMan();
		else
			return sslInstance;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(this.getClass().getSimpleName().toString());
		sb.append(" " + _dataType);
		sb.append(" " + _address.toString());
		sb.append(":" + _filepath);
		return sb.toString();
	}

	public static class FederatedRequestEncoder extends ObjectEncoder {
		@Override
		protected ByteBuf allocateBuffer(ChannelHandlerContext ctx, Serializable msg,
		boolean preferDirect) throws Exception {
			int initCapacity = 256; // default initial capacity
			if(msg instanceof FederatedRequest[]) {
				initCapacity = 0;
				try {
					for(FederatedRequest fr : (FederatedRequest[])msg) {
						int frSize = Math.toIntExact(fr.estimateSerializationBufferSize());
						if(Integer.MAX_VALUE - initCapacity < frSize) // summed sizes exceed integer limits
							throw new ArithmeticException("Overflow.");
						initCapacity += frSize;
					}
				} catch(ArithmeticException ae) { // size of federated request exceeds integer limits
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
