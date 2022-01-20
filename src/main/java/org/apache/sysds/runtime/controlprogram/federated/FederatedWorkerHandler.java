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
import java.io.InputStreamReader;
import java.net.InetSocketAddress;
import java.net.SocketAddress;
import java.util.Arrays;

import io.netty.channel.ChannelFuture;
import io.netty.channel.ChannelFutureListener;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.ChannelInboundHandlerAdapter;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.log4j.Logger;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.parser.DataExpression;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.BasicProgramBlock;
import org.apache.sysds.runtime.controlprogram.caching.CacheBlock;
import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.controlprogram.caching.FrameObject;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest.RequestType;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse.ResponseType;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.instructions.Instruction.IType;
import org.apache.sysds.runtime.instructions.InstructionParser;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.instructions.cp.ListObject;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.io.FileFormatPropertiesCSV;
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.runtime.lineage.LineageCache;
import org.apache.sysds.runtime.lineage.LineageCacheConfig;
import org.apache.sysds.runtime.lineage.LineageCacheConfig.ReuseCacheType;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.lineage.LineageItemUtils;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.meta.MetaDataAll;
import org.apache.sysds.runtime.meta.MetaDataFormat;
import org.apache.sysds.runtime.privacy.DMLPrivacyException;
import org.apache.sysds.runtime.privacy.PrivacyMonitor;
import org.apache.sysds.utils.Statistics;

/**
 * Note: federated worker handler created for every command; and concurrent parfor threads at coordinator need separate
 * execution contexts at the federated sites too
 */
public class FederatedWorkerHandler extends ChannelInboundHandlerAdapter {
	private static final Logger LOG = Logger.getLogger(FederatedWorkerHandler.class);

	private final FederatedLookupTable _flt;

	/**
	 * Create a Federated Worker Handler.
	 * 
	 * Note: federated worker handler created for every command; and concurrent parfor threads at coordinator need
	 * separate execution contexts at the federated sites too
	 * 
	 * @param flt The Federated Lookup Table of the current Federated Worker.
	 */
	public FederatedWorkerHandler(FederatedLookupTable flt) {
		_flt = flt;
	}

	@Override
	public void channelRead(ChannelHandlerContext ctx, Object msg) {
		ctx.writeAndFlush(createResponse(msg, ctx.channel().remoteAddress()))
			.addListener(new CloseListener());
	}

	protected FederatedResponse createResponse(Object msg) {
		return createResponse(msg, FederatedLookupTable.NOHOST);
	}

	private FederatedResponse createResponse(Object msg, SocketAddress remoteAddress) {
		String host;
		if(remoteAddress instanceof InetSocketAddress) {
			host = ((InetSocketAddress) remoteAddress).getHostString();
		}
		else if(remoteAddress instanceof SocketAddress) {
			host = remoteAddress.toString().split(":")[0].split("/")[1];
		}
		else {
			LOG.warn("Given remote address of coordinator is null. Continuing with "
				+ FederatedLookupTable.NOHOST + " as host identifier.");
			host = FederatedLookupTable.NOHOST;
		}

		return createResponse(msg, host);
	}

	private FederatedResponse createResponse(Object msg, String remoteHost) {
		if(!(msg instanceof FederatedRequest[]))
			return new FederatedResponse(ResponseType.ERROR,
				new FederatedWorkerHandlerException("Received object of wrong instance 'FederatedRequest[]'."));
		final FederatedRequest[] requests = (FederatedRequest[]) msg;
		try {
			return createResponse(requests, remoteHost);
		}
		catch(DMLPrivacyException | FederatedWorkerHandlerException ex) {
			// Here we control the error message, therefore it is allowed to send the stack trace with the response
			return new FederatedResponse(ResponseType.ERROR, ex);
		}
		catch(Exception ex) {
			// In all other cases it is not safe to send the exception message to the caller
			final String error = "Exception thrown while processing requests:\n" + Arrays.toString(requests);
			LOG.error(error, ex);
			return new FederatedResponse(ResponseType.ERROR, new FederatedWorkerHandlerException(error));
		}
	}

	private FederatedResponse createResponse(FederatedRequest[] requests, String remoteHost)
		throws DMLPrivacyException, FederatedWorkerHandlerException, Exception {
		FederatedResponse response = null; // last response
		boolean containsCLEAR = false;
		for(int i = 0; i < requests.length; i++) {
			final FederatedRequest request = requests[i];
			final RequestType t = request.getType();
			ExecutionContextMap ecm = _flt.getECM(remoteHost, request.getPID());
			logRequests(request, i, requests.length);

			PrivacyMonitor.setCheckPrivacy(request.checkPrivacy());
			PrivacyMonitor.clearCheckedConstraints();

			// execute command and handle privacy constraints
			final FederatedResponse tmp = executeCommand(request, ecm);
			conditionalAddCheckedConstraints(request, tmp);

			// select the response
			if(!tmp.isSuccessful()) {
				LOG.error("Command " + t + " resulted in error:\n" + tmp.getErrorMessage());
				return tmp; // Return first error without executing anything further
			}
			else if(t == RequestType.GET_VAR) {
				// If any of the requests was a GET_VAR then set it as output.
				if(response != null) {
					String message = "Multiple GET_VAR are not supported in single batch of requests.";
					LOG.error(message);
					throw new FederatedWorkerHandlerException(message);
				}
				response = tmp;
			}
			else if(response == null && i == requests.length - 1) {
				response = tmp; // return last
			}

			if(t == RequestType.CLEAR)
				containsCLEAR = true;
		}

		if(containsCLEAR)
			printStatistics();

		return response;
	}

	private static void printStatistics() {
		if(DMLScript.STATISTICS && Statistics.allowWorkerStatistics) {
			System.out.println("Federated Worker " + Statistics.display());
			Statistics.reset();
		}
	}

	private static void logRequests(FederatedRequest request, int nrRequest, int totalRequests) {
		if(LOG.isDebugEnabled()) {
			LOG.debug("Executing command " + (nrRequest + 1) + "/" + totalRequests + ": " + request.getType().name());
			if(LOG.isTraceEnabled()) 
				LOG.trace("full command: " + request.toString());
		}
	}

	private static void conditionalAddCheckedConstraints(FederatedRequest request, FederatedResponse response) {
		if(request.checkPrivacy())
			response.setCheckedConstraints(PrivacyMonitor.getCheckedConstraints());
	}

	private FederatedResponse executeCommand(FederatedRequest request, ExecutionContextMap ecm)
		throws DMLPrivacyException, FederatedWorkerHandlerException, Exception {
		final RequestType method = request.getType();
		switch(method) {
			case READ_VAR:
				return readData(request, ecm); // matrix/frame
			case PUT_VAR:
				return putVariable(request, ecm);
			case GET_VAR:
				return getVariable(request, ecm);
			case EXEC_INST:
				return execInstruction(request, ecm);
			case EXEC_UDF:
				return execUDF(request, ecm);
			case CLEAR:
				return execClear(ecm);
			case NOOP:
				return execNoop();
			default:
				String message = String.format("Method %s is not supported.", method);
				throw new FederatedWorkerHandlerException(message);
		}
	}

	private FederatedResponse readData(FederatedRequest request, ExecutionContextMap ecm) {
		checkNumParams(request.getNumParams(), 2);
		String filename = (String) request.getParam(0);
		DataType dt = DataType.valueOf((String) request.getParam(1));
		return readData(filename, dt, request.getID(), request.getTID(), ecm);
	}

	private FederatedResponse readData(String filename, Types.DataType dataType,
		long id, long tid, ExecutionContextMap ecm) {
		MatrixCharacteristics mc = new MatrixCharacteristics();
		mc.setBlocksize(ConfigurationManager.getBlocksize());
		CacheableData<?> cd;
		switch(dataType) {
			case MATRIX:
				cd = new MatrixObject(Types.ValueType.FP64, filename);
				break;
			case FRAME:
				cd = new FrameObject(filename);
				break;
			default:
				throw new FederatedWorkerHandlerException("Could not recognize datatype");
		}

		FileFormat fmt = null;
		boolean header = false;
		String delim = null;
		FileSystem fs = null;
		MetaDataAll mtd;

		try {
			final String mtdName = DataExpression.getMTDFileName(filename);
			Path path = new Path(mtdName);
			fs = IOUtilFunctions.getFileSystem(mtdName);
			try(BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(path)))) {
				mtd = new MetaDataAll(br);
				if(!mtd.mtdExists())
					throw new FederatedWorkerHandlerException("Could not parse metadata file");
				mc.setRows(mtd.getDim1());
				mc.setCols(mtd.getDim2());
				mc.setNonZeros(mtd.getNnz());
				header = mtd.getHasHeader();
				cd = mtd.parseAndSetPrivacyConstraint(cd);
				fmt = mtd.getFileFormat();
				delim = mtd.getDelim();
			}
		}
		catch(DMLPrivacyException | FederatedWorkerHandlerException ex) {
			throw ex;
		}
		catch(Exception ex) {
			String msg = "Exception of type " + ex.getClass() + " thrown when processing READ request";
			LOG.error(msg, ex);
			throw new DMLRuntimeException(msg);
		}
		finally {
			IOUtilFunctions.closeSilently(fs);
		}

		// put meta data object in symbol table, read on first operation
		cd.setMetaData(new MetaDataFormat(mc, fmt));
		if(fmt == FileFormat.CSV)
			cd.setFileFormatProperties(new FileFormatPropertiesCSV(header, delim, DataExpression.DEFAULT_DELIM_SPARSE));
		cd.enableCleanup(false); // guard against deletion
		ecm.get(tid).setVariable(String.valueOf(id), cd);

		if(DMLScript.LINEAGE)
			// create a literal type lineage item with the file name
			ecm.get(tid).getLineage().set(String.valueOf(id), new LineageItem(filename));

		if(dataType == Types.DataType.FRAME) {
			FrameObject frameObject = (FrameObject) cd;
			frameObject.acquireRead();
			frameObject.refreshMetaData(); // get block schema
			frameObject.release();
			return new FederatedResponse(ResponseType.SUCCESS, new Object[] {id, frameObject.getSchema(), mc});
		}
		return new FederatedResponse(ResponseType.SUCCESS, new Object[] {id, mc});
	}

	private FederatedResponse putVariable(FederatedRequest request, ExecutionContextMap ecm) {
		checkNumParams(request.getNumParams(), 1, 2);
		final String varName = String.valueOf(request.getID());
		ExecutionContext ec = ecm.get(request.getTID());

		if(ec.containsVariable(varName)) {
			Data tgtData = ec.removeVariable(varName);
			if(tgtData != null)
				ec.cleanupDataObject(tgtData);
			LOG.warn("Variable" + request.getID() + " already existing, fallback to overwritten.");
		}

		// wrap transferred cache block into cacheable data
		Data data;
		if(request.getParam(0) instanceof CacheBlock)
			data = ExecutionContext.createCacheableData((CacheBlock) request.getParam(0));
		else if(request.getParam(0) instanceof ScalarObject)
			data = (ScalarObject) request.getParam(0);
		else if(request.getParam(0) instanceof ListObject)
			data = (ListObject) request.getParam(0);
		else if(request.getNumParams() == 2)
			data = request.getParam(1) == DataType.MATRIX ? ExecutionContext
				.createMatrixObject((MatrixCharacteristics) request.getParam(0)) : ExecutionContext
					.createFrameObject((MatrixCharacteristics) request.getParam(0));
		else
			throw new FederatedWorkerHandlerException(
				"Unsupported object type, has to be of type CacheBlock or ScalarObject");

		// set variable and construct empty response
		ec.setVariable(varName, data);
		if(DMLScript.LINEAGE && request.getNumParams()==1)
			// don't trace if the data contains only metadata
			ec.getLineage().set(varName, new LineageItem(String.valueOf(request.getChecksum(0))));

		return new FederatedResponse(ResponseType.SUCCESS_EMPTY);
	}

	private FederatedResponse getVariable(FederatedRequest request, ExecutionContextMap ecm) {
		checkNumParams(request.getNumParams(), 0);
		ExecutionContext ec = ecm.get(request.getTID());
		if(!ec.containsVariable(String.valueOf(request.getID())))
			throw new FederatedWorkerHandlerException(
				"Variable " + request.getID() + " does not exist at federated worker.");

		// get variable and construct response
		Data dataObject = ec.getVariable(String.valueOf(request.getID()));
		dataObject = PrivacyMonitor.handlePrivacy(dataObject);
		switch(dataObject.getDataType()) {
			case TENSOR:
			case MATRIX:
			case FRAME:
				return new FederatedResponse(ResponseType.SUCCESS, ((CacheableData<?>) dataObject).acquireReadAndRelease());
			case LIST:
				return new FederatedResponse(ResponseType.SUCCESS, ((ListObject) dataObject).getData());
			case SCALAR:
				return new FederatedResponse(ResponseType.SUCCESS, dataObject);
			default:
				throw new FederatedWorkerHandlerException("Unsupported return datatype " + dataObject.getDataType().name());
		}
	}

	private FederatedResponse execInstruction(FederatedRequest request, ExecutionContextMap ecm) throws Exception {
		ExecutionContext ec = ecm.get(request.getTID());
		
		//handle missing spark execution context
		//TODO handling of spark instructions should be under control of federated site not coordinator
		Instruction receivedInstruction = InstructionParser.parseSingleInstruction((String) request.getParam(0));
		if(receivedInstruction.getType() == IType.SPARK
			&& !(ec instanceof SparkExecutionContext) ) {
			ecm.convertToSparkCtx();
			ec = ecm.get(request.getTID());
		}

		BasicProgramBlock pb = new BasicProgramBlock(null);
		pb.getInstructions().clear();
		pb.getInstructions().add(receivedInstruction);

		if(DMLScript.LINEAGE)
			// Compiler assisted optimizations are not applicable for Fed workers.
			// e.g. isMarkedForCaching fails as output operands are saved in the
			// symbol table only after the instruction execution finishes.
			// NOTE: In shared JVM, this will disable compiler assistance even for the coordinator
			LineageCacheConfig.setCompAssRW(false);

		pb.execute(ec); // execute single instruction

		return new FederatedResponse(ResponseType.SUCCESS_EMPTY);
	}

	private FederatedResponse execUDF(FederatedRequest request, ExecutionContextMap ecm) {
		checkNumParams(request.getNumParams(), 1);
		ExecutionContext ec = ecm.get(request.getTID());

		// get function and input parameters
		FederatedUDF udf = (FederatedUDF) request.getParam(0);
		Data[] inputs = Arrays.stream(udf.getInputIDs()).mapToObj(id -> ec.getVariable(String.valueOf(id)))
			.map(PrivacyMonitor::handlePrivacy).toArray(Data[]::new);

		// trace lineage
		if(DMLScript.LINEAGE)
			LineageItemUtils.traceFedUDF(ec, udf);

		// reuse or execute user-defined function
		try {
			// reuse UDF outputs if available in lineage cache
			FederatedResponse reuse = LineageCache.reuse(udf, ec);
			if(reuse.isSuccessful())
				return reuse;

			// else execute the UDF
			long t0 = !ReuseCacheType.isNone() ? System.nanoTime() : 0;
			FederatedResponse res = udf.execute(ec, inputs);
			long t1 = !ReuseCacheType.isNone() ? System.nanoTime() : 0;
			// cacheUDFOutputs(udf, inputs, t1-t0, ec);
			LineageCache.putValue(udf, ec, t1 - t0);
			return res;
		}
		catch(DMLPrivacyException | FederatedWorkerHandlerException ex) {
			throw ex;
		}
		catch(Exception ex) {
			// Note it is unsafe to throw the ex trace along with the exception here.
			String msg = "Exception of type " + ex.getClass() + " thrown when processing EXEC_UDF request";
			LOG.error(msg, ex);
			throw new FederatedWorkerHandlerException(msg);
		}
	}

	private FederatedResponse execClear(ExecutionContextMap ecm) {
		try {
			ecm.clear();
		}
		catch(DMLPrivacyException | FederatedWorkerHandlerException ex) {
			throw ex;
		}
		catch(Exception ex) {
			String msg = "Exception of type " + ex.getClass() + " thrown when processing CLEAR request";
			LOG.error(msg, ex);
			throw new FederatedWorkerHandlerException(msg);
		}
		return new FederatedResponse(ResponseType.SUCCESS_EMPTY);
	}

	private static FederatedResponse execNoop() {
		return new FederatedResponse(ResponseType.SUCCESS_EMPTY);
	}

	private static void checkNumParams(int actual, int... expected) {
		if(Arrays.stream(expected).anyMatch(x -> x == actual))
			return;
		throw new DMLRuntimeException("FederatedWorkerHandler: Received wrong amount of params:" + " expected="
			+ Arrays.toString(expected) + ", actual=" + actual);
	}

	@Override
	public void exceptionCaught(ChannelHandlerContext ctx, Throwable cause) {
		cause.printStackTrace();
		ctx.close();
	}

	private static class CloseListener implements ChannelFutureListener {
		@Override
		public void operationComplete(ChannelFuture channelFuture) throws InterruptedException {
			if(!channelFuture.isSuccess()) {
				LOG.error("Federated Worker Write failed");
				channelFuture.channel().writeAndFlush(new FederatedResponse(ResponseType.ERROR,
					new FederatedWorkerHandlerException("Error while sending response."))).channel().close().sync();
			}
			else {
				PrivacyMonitor.clearCheckedConstraints();
				channelFuture.channel().close().sync();
			}
		}
	}
}
