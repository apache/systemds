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

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Serializable;
import java.net.ConnectException;
import java.net.InetSocketAddress;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.Future;

import io.netty.bootstrap.Bootstrap;
import io.netty.channel.ChannelFuture;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.ChannelInboundHandlerAdapter;
import io.netty.channel.ChannelInitializer;
import io.netty.channel.ChannelOutboundHandlerAdapter;
import io.netty.channel.ChannelPipeline;
import io.netty.channel.EventLoopGroup;
import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.sysds.common.Types;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.parser.DataExpression;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.meta.MetaData;
import org.apache.sysds.runtime.meta.MetaDataAll;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest.RequestType;
import org.apache.sysds.runtime.controlprogram.paramserv.NetworkTrafficCounter;
import org.apache.sysds.runtime.controlprogram.caching.CacheBlock;
import org.apache.sysds.conf.DMLConfig;
import io.netty.buffer.ByteBuf;
import io.netty.channel.nio.NioEventLoopGroup;
import io.netty.channel.socket.SocketChannel;
import io.netty.channel.socket.nio.NioSocketChannel;
import io.netty.handler.codec.serialization.ObjectEncoder;
import io.netty.handler.timeout.ReadTimeoutHandler;
import io.netty.util.concurrent.Promise;

@SuppressWarnings("deprecation")
public class FederatedData {
	private static final Log LOG = LogFactory.getLog(FederatedData.class.getName());
	private static final Set<InetSocketAddress> _allFedSites = new HashSet<>();

	/** Thread pool specific for the federated requests */
	private static EventLoopGroup workerGroup = null;



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
		FederatedRequest request = (mtd != null) ? new FederatedRequest(RequestType.READ_VAR, id,
			mtd) : new FederatedRequest(RequestType.READ_VAR, id);
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
	public static Future<FederatedResponse> executeFederatedOperation(InetSocketAddress address,
		FederatedRequest... request) {
		return executeFederatedOperation(address, 1, request);
	}

	/**
	 * Executes an federated operation on a federated worker.
	 *
	 * @param address socket address (incl host and port)
	 * @param retry   the retry count
	 * @param request the requested operation
	 * @return the response
	 */
	public synchronized static Future<FederatedResponse> executeFederatedOperation(InetSocketAddress address, int retry,
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
			if(e instanceof ConnectException) {

				if(retry < 5) {
					try {
						// Increasing retry timeout
						Thread.sleep(200 * retry);
					}
					catch(Exception e2) {
						throw new DMLRuntimeException(e);
					}
					return executeFederatedOperation(address, retry + 1, request);
				}
				else {
					throw new DMLRuntimeException(e);
				}
			}
			throw new DMLRuntimeException("Failed sending federated operation", e);
		}
	}

	private static ChannelInitializer<SocketChannel> createChannel(InetSocketAddress address,
		DataRequestHandler handler) {
		final int timeout = ConfigurationManager.getFederatedTimeout();
		final boolean ssl = ConfigurationManager.isFederatedSSL();

		return new ChannelInitializer<>() {
			@Override
			protected void initChannel(SocketChannel ch) throws Exception {
				final ChannelPipeline cp = ch.pipeline();
				final Optional<ImmutablePair<ChannelInboundHandlerAdapter, ChannelOutboundHandlerAdapter>> compressionStrategy = FederationUtils.compressionStrategy();
				cp.addLast("NetworkTrafficCounter", new NetworkTrafficCounter(FederatedStatistics::logServerTraffic));

				if(ssl)
					cp.addLast(FederatedSSLUtil.createSSLHandler(ch, address));
				if(timeout > -1)
					cp.addLast(new ReadTimeoutHandler(timeout));

				compressionStrategy.ifPresent(strategy -> cp.addLast(strategy.left));
				cp.addLast(FederationUtils.decoder());
				compressionStrategy.ifPresent(strategy -> cp.addLast(strategy.right));
				cp.addLast(new FederatedRequestEncoder());
				cp.addLast(handler);
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
		protected ByteBuf allocateBuffer(ChannelHandlerContext ctx, Serializable msg, boolean preferDirect)
			throws Exception {
			int initCapacity = 256; // default initial capacity
			if(msg instanceof FederatedRequest[]) {
				initCapacity = 0;
				try {
					for(FederatedRequest fr : (FederatedRequest[]) msg) {
						int frSize = Math.toIntExact(fr.estimateSerializationBufferSize());
						if(Integer.MAX_VALUE - initCapacity < frSize) // summed sizes exceed integer limits
							throw new ArithmeticException("Overflow.");
						initCapacity += frSize;
					}
				}
				catch(ArithmeticException ae) { // size of federated request exceeds integer limits
					initCapacity = Integer.MAX_VALUE;
				}
			}
			if(preferDirect)
				return ctx.alloc().ioBuffer(initCapacity);
			else
				return ctx.alloc().heapBuffer(initCapacity);
		}
	}

	/**
	 * Requests privacy constraints from the federated worker
	 * 
	 * @return Future containing the federated response with privacy constraints
	 */
	public Future<FederatedResponse> requestPrivacyConstraints() {
		if (!isInitialized())
			throw new DMLRuntimeException("Cannot request privacy constraints from uninitialized federated data");
		
		FederatedRequest request = new FederatedRequest(RequestType.EXEC_UDF, _varID, new GetPrivacyConstraints(_filepath));
		return executeFederatedOperation(request);
	}

	public static class GetPrivacyConstraints extends FederatedUDF {
		private static final long serialVersionUID = 1637852940793579590L;
		private final String filename;

		public GetPrivacyConstraints(String filename) {
			super(new long[] { });  // Pass empty ID array to parent constructor as this is a static class
			this.filename = filename;
		}
	
		@Override
		public FederatedResponse execute(ExecutionContext ec, Data... data) {
			String privacyConstraints = null;
			FileSystem fs = null;
			MetaDataAll mtd = null;
		
			try {
				final String mtdName = DataExpression.getMTDFileName(filename);
				Path path = new Path(mtdName);
				fs = IOUtilFunctions.getFileSystem(mtdName);
				try(BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(path)))) {
					mtd = new MetaDataAll(br);
					if(!mtd.mtdExists())
						throw new FederatedWorkerHandlerException("Could not parse metadata file for " + filename);
					privacyConstraints = mtd.getPrivacyConstraints();
					
					if(privacyConstraints == null)
						LOG.warn("No privacy constraints found in metadata for " + filename);
				}
				
				return new FederatedResponse(FederatedResponse.ResponseType.SUCCESS, privacyConstraints);
			}
			catch(IOException ex) {
				String msg = "IO Exception when reading metadata file for " + filename;
				LOG.error(msg, ex);
				throw new FederatedWorkerHandlerException(msg, ex);
			}
			catch(Exception ex) {
				String msg = "Exception of type " + ex.getClass() + " thrown when processing privacy constraints request for " + filename;
				LOG.error(msg, ex);
				throw new FederatedWorkerHandlerException(msg, ex);
			}
			finally {
				IOUtilFunctions.closeSilently(fs);
			}
		}

		@Override
		public Pair<String, LineageItem> getLineageItem(ExecutionContext ec) {
			String opcode = "fedprivconst"; // Appropriate operation code
			
			// Create input LineageItem for the operation
			LineageItem[] inputs = new LineageItem[] { 
				new LineageItem(filename) // Create literal LineageItem by passing only the string
			};
			
			// Create appropriate LineageItem (for read operation)
			return Pair.of(opcode, new LineageItem(opcode, inputs));
		}
	}
}