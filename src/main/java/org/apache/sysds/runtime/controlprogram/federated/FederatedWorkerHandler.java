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
import java.time.LocalDateTime;
import java.util.Arrays;
import java.util.concurrent.CompletableFuture;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.lops.Compression.CompressConfig;
import org.apache.sysds.parser.DataExpression;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressedMatrixBlockFactory;
import org.apache.sysds.runtime.controlprogram.BasicProgramBlock;
import org.apache.sysds.runtime.controlprogram.caching.CacheBlock;
import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.controlprogram.caching.FrameObject;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest.RequestType;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse.ResponseType;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.DataObjectModel;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.EventModel;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.EventStageModel;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.RequestModel;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.instructions.Instruction.IType;
import org.apache.sysds.runtime.instructions.InstructionParser;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.ComputationCPInstruction;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.instructions.cp.ListObject;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.io.FileFormatPropertiesCSV;
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.runtime.lineage.Lineage;
import org.apache.sysds.runtime.lineage.LineageCache;
import org.apache.sysds.runtime.lineage.LineageCacheConfig;
import org.apache.sysds.runtime.lineage.LineageCacheConfig.ReuseCacheType;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.lineage.LineageItemUtils;
import org.apache.sysds.runtime.matrix.operators.MultiThreadedOperator;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.meta.MetaDataAll;
import org.apache.sysds.runtime.meta.MetaDataFormat;
import org.apache.sysds.utils.Statistics;
import org.apache.sysds.utils.stats.InfrastructureAnalyzer;
import org.apache.sysds.utils.stats.ParamServStatistics;
import org.apache.sysds.utils.stats.Timing;

import io.netty.channel.ChannelFuture;
import io.netty.channel.ChannelFutureListener;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.ChannelInboundHandlerAdapter;

/**
 * Note: federated worker handler created for every command; and concurrent parfor threads at coordinator need separate
 * execution contexts at the federated sites too
 */
public class FederatedWorkerHandler extends ChannelInboundHandlerAdapter {
	private static final Log LOG = LogFactory.getLog(FederatedWorkerHandler.class.getName());

	/** The Federated Lookup Table of the current Federated Worker. */
	private final FederatedLookupTable _flt;

	/** Read cache shared by all worker handlers */
	private final FederatedReadCache _frc;
	private Timing _timing = null;
	
	/** Federated workload analyzer */
	private final FederatedWorkloadAnalyzer _fan;

	private String _remoteAddress = FederatedLookupTable.NOHOST;

	/**
	 * Create a Federated Worker Handler.
	 * 
	 * Note: federated worker handler created for every command; and concurrent parfor threads at coordinator need
	 * separate execution contexts at the federated sites too
	 * 
	 * @param flt The Federated Lookup Table of the current Federated Worker.
	 * @param frc Read cache shared by all worker handlers.
	 * @param fan A Workload analyzer object (should be null if not used).
	 */
	public FederatedWorkerHandler(FederatedLookupTable flt, FederatedReadCache frc, FederatedWorkloadAnalyzer fan) {
		_flt = flt;
		_frc = frc;
		_fan = fan;
		
		if(DMLScript.LINEAGE) {
			// Compiler assisted optimizations are not applicable for Fed workers.
			// e.g. isMarkedForCaching fails as output operands are saved in the
			// symbol table only after the instruction execution finishes.
			// NOTE: In shared JVM, this will disable compiler assistance even for the coordinator
			LineageCacheConfig.setCompAssRW(false);
		}
	}
	
	public FederatedWorkerHandler(FederatedLookupTable flt, FederatedReadCache frc, FederatedWorkloadAnalyzer fan, Timing timing) {
		this(flt, frc, fan);
		_timing = timing;
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
		try {
			if (_timing != null) {
				ParamServStatistics.accFedNetworkTime((long) _timing.stop());
			}
		} catch (RuntimeException ignored) {
			// ignore timing if it wasn't started yet
		}
		
		String host;
		if(remoteAddress == null) {
			LOG.warn("Given remote address of coordinator is null. Continuing with "
				+ FederatedLookupTable.NOHOST + " as host identifier.");
			host = FederatedLookupTable.NOHOST;
		}
		else if(remoteAddress instanceof InetSocketAddress) {
			host = ((InetSocketAddress) remoteAddress).getHostString();
			_remoteAddress = remoteAddress.toString();
		}
		else {
			host = remoteAddress.toString().split(":")[0].split("/")[1];
			_remoteAddress = remoteAddress.toString();
		}
		

		FederatedResponse res = createResponse(msg, host);
		if (_timing != null) {
			_timing.start();
		}
		return res;
	}

	private FederatedResponse createResponse(Object msg, String remoteHost) {
		if(!(msg instanceof FederatedRequest[]))
			return new FederatedResponse(ResponseType.ERROR,
				new FederatedWorkerHandlerException("Received object of wrong instance 'FederatedRequest[]'."));
		final FederatedRequest[] requests = (FederatedRequest[]) msg;
		try {
			return createResponse(requests, remoteHost);
		}
		catch(FederatedWorkerHandlerException ex) {
			// Here we control the error message, therefore it is allowed to send the stack trace with the response
			LOG.error("Exception in FederatedWorkerHandler while processing requests:\n"
				+ Arrays.toString(requests), ex);
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
		throws FederatedWorkerHandlerException, Exception {
			
		FederatedResponse response = null; // last response
		boolean containsCLEAR = false;
		long clearReqPid = -1;
		int numGETrequests = 0;
		var event = new EventModel();
		final String coordinatorHostIdFormat = "%s-%d";
		event.setCoordinatorHostId(String.format(coordinatorHostIdFormat, remoteHost, requests[0].getPID()));
		for(int i = 0; i < requests.length; i++) {
			final FederatedRequest request = requests[i];
			final RequestType t = request.getType();
			final ExecutionContextMap ecm = _flt.getECM(remoteHost, request.getPID());
			logRequests(request, i, requests.length);

			var eventStage = new EventStageModel();
			// execute command and handle privacy constraints
			final FederatedResponse tmp = executeCommand(request, ecm, eventStage);

			if (DMLScript.STATISTICS) {
				var requestStat = new RequestModel(request.getType().name(), 1L);
				requestStat.setCoordinatorHostId(String.format(coordinatorHostIdFormat, remoteHost, request.getPID()));
				FederatedStatistics.addWorkerRequest(requestStat);

				event.stages.add(eventStage);
			}

			// select the response
			if(!tmp.isSuccessful()) {
				LOG.error("Command " + t + " resulted in error:\n" + tmp.getErrorMessage());
				if (DMLScript.STATISTICS)
					FederatedStatistics.addEvent(event);
				return tmp; // Return first error without executing anything further
			}
			else if(t == RequestType.GET_VAR) {
				// If any of the requests was a GET_VAR then set it as output.
				if(response != null && numGETrequests > 0) {
					String message = "Multiple GET_VAR are not supported in single batch of requests.";
					LOG.error(message);
					if (DMLScript.STATISTICS)
						FederatedStatistics.addEvent(event);
					throw new FederatedWorkerHandlerException(message);
				}
				response = tmp;
				numGETrequests ++;
			}
			else if(response == null
				&& (t == RequestType.EXEC_INST || t == RequestType.EXEC_UDF)) {
				// If there was no GET, use the EXEC INST or UDF to obtain the returned nnz
				response = tmp;
			}
			else if(response == null && i == requests.length - 1) {
				response = tmp; // return last
			}

			if (DMLScript.STATISTICS) {
				if(t == RequestType.PUT_VAR || t == RequestType.EXEC_UDF) {
					for (int paramIndex = 0; paramIndex < request.getNumParams(); paramIndex++)
						FederatedStatistics.incFedTransfer(request.getParam(paramIndex), _remoteAddress, request.getPID());
				}
				if(t == RequestType.GET_VAR) {
					var data = response.getData();
					for (int dataObjIndex = 0; dataObjIndex < Arrays.stream(data).count(); dataObjIndex++)
						FederatedStatistics.incFedTransfer(data[dataObjIndex], _remoteAddress, request.getPID());
				}
			}

			if(t == RequestType.CLEAR) {
				containsCLEAR = true;
				clearReqPid = request.getPID();
			}
		}

		if(containsCLEAR) {
			_flt.removeECM(remoteHost, clearReqPid);
			printStatistics();
		}

		if (DMLScript.STATISTICS)
			FederatedStatistics.addEvent(event);

		return response;
	}

	private static void printStatistics() {
		if(DMLScript.STATISTICS && Statistics.allowWorkerStatistics) {
			System.out.println("Federated Worker " + Statistics.display());
			// Statistics.reset();
		}
	}

	private static void logRequests(FederatedRequest request, int nrRequest, int totalRequests) {
		if(LOG.isDebugEnabled()) {
			LOG.debug("Executing command " + (nrRequest + 1) + "/" + totalRequests + ": " + request.getType().name());
			if(LOG.isTraceEnabled()) 
				LOG.trace("full command: " + request.toString());
		}
	}

	private FederatedResponse executeCommand(FederatedRequest request, ExecutionContextMap ecm, EventStageModel eventStage)
		throws FederatedWorkerHandlerException, Exception
	{
		final RequestType method = request.getType();
		FederatedResponse result = null;

		eventStage.startTime = LocalDateTime.now();

		switch(method) {
			case READ_VAR:
				eventStage.operation = method.name();
				result = readData(request, ecm); // matrix/frame
				break;
			case PUT_VAR:
				eventStage.operation = method.name();
				result = putVariable(request, ecm);
				break;
			case GET_VAR:
				eventStage.operation = method.name();
				result = getVariable(request, ecm);
				break;
			case EXEC_INST:
				result = execInstruction(request, ecm, eventStage);
				break;
			case EXEC_UDF:
				result = execUDF(request, ecm, eventStage);
				break;
			case CLEAR:
				eventStage.operation = method.name();
				result = execClear(ecm);
				break;
			case NOOP:
				eventStage.operation = method.name();
				result = execNoop();
				break;
			default:
				String message = String.format("Method %s is not supported.", method);
				throw new FederatedWorkerHandlerException(message);
		}

		eventStage.endTime = LocalDateTime.now();

		return result;
	}

	private FederatedResponse readData(FederatedRequest request, ExecutionContextMap ecm) {
		checkNumParams(request.getNumParams(), 2, 3);
		String filename = (String) request.getParam(0);
		DataType dt = DataType.valueOf((String) request.getParam(1));
		return readData(filename, dt, request.getID(), request.getTID(), ecm,
			request.getNumParams() == 2 ? null : (CacheBlock<?>)request.getParam(2));
	}

	private FederatedResponse readData(String filename, DataType dataType,
		long id, long tid, ExecutionContextMap ecm, CacheBlock<?> localBlock) {
		MatrixCharacteristics mc = new MatrixCharacteristics();
		mc.setBlocksize(ConfigurationManager.getBlocksize());

		if(dataType != DataType.MATRIX && dataType != DataType.FRAME)
			// early throwing of exception to avoid infinitely waiting threads for data
			throw new FederatedWorkerHandlerException("Could not recognize datatype");

		final ExecutionContext ec = ecm.get(tid);
		final LineageItem linItem = new LineageItem(filename);
		CacheableData<?> cd = null;
		final String sId = String.valueOf(id);

		boolean linReuse = (!ReuseCacheType.isNone() && dataType == DataType.MATRIX);
		boolean readCache = ConfigurationManager.isFederatedReadCacheEnabled();
		if(!linReuse || !LineageCache.reuseFedRead(sId, dataType, linItem, ec)) {
			// Lookup read cache if reuse is disabled and we skipped storing in the
			// lineage cache due to other constraints
			cd = _frc.get(filename, readCache & !linReuse);
			try {
				if(cd == null) { // data is neither in lineage cache nor in read cache
					cd = localBlock == null ?
						readDataNoReuse(filename, dataType, mc) :
						ExecutionContext.createCacheableData(localBlock); // actual read of the data
					if(linReuse) // put the object into the lineage cache
						LineageCache.putFedReadObject(cd, linItem, ec);
					else if( readCache )
						_frc.setData(filename, cd); // set the data into the read cache entry
				}
				ec.setVariable(sId, cd);

			} catch(Exception ex) {
				if(linReuse)
					LineageCache.putFedReadObject(null, linItem, ec); // removing the placeholder
				else
					_frc.setInvalid(filename);
				throw ex;
			}
		}
		
		if(shouldTryAsyncCompress()) // TODO: replace the reused object
			CompressedMatrixBlockFactory.compressAsync(ec, sId);

		if(DMLScript.LINEAGE)
			// create a literal type lineage item with the file name
			ec.getLineage().set(sId, linItem);

		if(dataType == Types.DataType.FRAME) { // frame read
			FrameObject frameObject = (FrameObject) cd;
			if(frameObject == null)
				return new FederatedResponse(ResponseType.ERROR);
			else{
				frameObject.acquireRead();
				frameObject.refreshMetaData(); // get block schema
				frameObject.release();
				return new FederatedResponse(ResponseType.SUCCESS, new Object[] {id, frameObject.getSchema(), mc});
			}
		}
		else // matrix read
			return new FederatedResponse(ResponseType.SUCCESS, new Object[] {id, mc});
	}

	private CacheableData<?> readDataNoReuse(String filename, DataType dataType,
		MatrixCharacteristics mc) {
		CacheableData<?> cd = null;

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
				fmt = mtd.getFileFormat();
				delim = mtd.getDelim();
			}
		}
		catch(FederatedWorkerHandlerException ex) {
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
			cd.setFileFormatProperties(new FileFormatPropertiesCSV(header, delim,
				DataExpression.DEFAULT_DELIM_SPARSE));
		cd.enableCleanup(false); // guard against deletion

		return cd;
	}

	private FederatedResponse putVariable(FederatedRequest request, ExecutionContextMap ecm) {
		checkNumParams(request.getNumParams(), 1, 2);
		final String varName = String.valueOf(request.getID());
		ExecutionContext ec = ecm.get(request.getTID());
		
		if(ec.containsVariable(varName)) {
			final Data tgtData = ec.removeVariable(varName);
			if(tgtData != null)
				ec.cleanupDataObject(tgtData);
			LOG.warn("Variable" + request.getID() + " already existing, fallback to overwritten.");
		}

		final Object v = request.getParam(0);
		// wrap transferred cache block into cacheable data
		Data data;
		long size = 0;
		if(v instanceof CacheBlock) {
			var block = ExecutionContext.createCacheableData((CacheBlock<?>) v);
			size = block.getDataSize();
			data = block;
		}
		else if(v instanceof ScalarObject) {
			data = (ScalarObject) v;
			size = ((ScalarObject) v).getSize();
		}
		else if(v instanceof ListObject) {
			data = (ListObject) v;
			size = ((ListObject) v).getDataSize();
		}
		else if(request.getNumParams() == 2){
			final Object v1= request.getParam(1);
			if(v1 == DataType.MATRIX) {
				var mtrx = ExecutionContext.createMatrixObject((MatrixCharacteristics) v);
				size = mtrx.getDataSize();
				data = mtrx;
			}
			else {
				var frm = ExecutionContext.createFrameObject((MatrixCharacteristics) v);
				size = frm.getDataSize();
				data = frm;
			}
		}
		else
			throw new FederatedWorkerHandlerException(
				"Unsupported object type, has to be of type CacheBlock or ScalarObject");

				
		// set variable and construct empty response
		ec.setVariable(varName, data);

		if (DMLScript.STATISTICS){
			FederatedStatistics.addDataObject(new DataObjectModel(varName, data.getDataType().name(), data.getValueType().name(), size));
		}

		if(shouldTryAsyncCompress())
			CompressedMatrixBlockFactory.compressAsync(ec, varName);

		if(DMLScript.LINEAGE) {
			if(request.getParam(0) instanceof CacheBlock && request.getLineageTrace() != null) {
				ec.getLineage().set(varName, Lineage.deserializeSingleTrace(request.getLineageTrace()));
				if(DMLScript.STATISTICS)
					FederatedStatistics.aggFedPutLineage(request.getLineageTrace());
			}
			else if(request.getParam(0) instanceof ScalarObject)
				ec.getLineage().set(varName, new LineageItem(CPOperand.getLineageLiteral((ScalarObject)request.getParam(0), true)));
			else if(request.getNumParams()==1) // don't trace if the data contains only metadata
				ec.getLineage().set(varName, new LineageItem(String.valueOf(request.getChecksum(0))));
		}

		return new FederatedResponse(ResponseType.SUCCESS_EMPTY);
	}

	private FederatedResponse getVariable(FederatedRequest request, ExecutionContextMap ecm) {
		try{
			checkNumParams(request.getNumParams(), 0);
			ExecutionContext ec = ecm.get(request.getTID());
			if(!ec.containsVariable(String.valueOf(request.getID())))
				throw new FederatedWorkerHandlerException(
					"Variable " + request.getID() + " does not exist at federated worker.");
	
			// get variable and construct response
			Data dataObject = ec.getVariable(String.valueOf(request.getID()));
			switch(dataObject.getDataType()) {
				case TENSOR:
				case MATRIX:
				case FRAME:
					return new FederatedResponse(ResponseType.SUCCESS, ((CacheableData<?>) dataObject).acquireReadAndRelease(),
						ReuseCacheType.isNone() ? null : ec.getLineage().get(String.valueOf(request.getID())));
				case LIST:
					return new FederatedResponse(ResponseType.SUCCESS, ((ListObject) dataObject).getData());
				case SCALAR:
					return new FederatedResponse(ResponseType.SUCCESS, dataObject);
				default:
					throw new FederatedWorkerHandlerException("Unsupported return datatype " + dataObject.getDataType().name());
			}
		}
		catch(Exception e){
			throw new FederatedWorkerHandlerException("Failed to getVariable " , e);
		}
	}

	private FederatedResponse execInstruction(FederatedRequest request, ExecutionContextMap ecm, EventStageModel eventStage) throws Exception {
		final Instruction ins = InstructionParser.parseSingleInstruction((String) request.getParam(0));

		eventStage.operation = ins.getExtendedOpcode();

		final long tid = request.getTID();
		final ExecutionContext ec = getContextForInstruction(tid, ins, ecm);
		setThreads(ins);
		exec(ec, ins);
		adaptToWorkload(ec, _fan, tid, ins);
		return new FederatedResponse(
			ResponseType.SUCCESS_EMPTY, getOutputNnz(ec, ins));
	}
	
	private static ExecutionContext getContextForInstruction(long id, Instruction ins, ExecutionContextMap ecm){
		final ExecutionContext ec = ecm.get(id);
		//handle missing spark execution context
		//TODO handling of spark instructions should be under control of federated site not coordinator
		if(ins.getType() == IType.SPARK
			&& !(ec instanceof SparkExecutionContext) )
		{
			ecm.convertToSparkCtx();
			return ecm.get(id);
		}
		return ec;
	}

	private static void setThreads(Instruction ins){
		final Operator op = ins.getOperator();
		if(op instanceof MultiThreadedOperator) {
			final int par_inst = ConfigurationManager.getDMLConfig().getIntValue(DMLConfig.FEDERATED_PAR_INST);
			final int k = (par_inst > 0) ? par_inst : InfrastructureAnalyzer.getLocalParallelism();
			((MultiThreadedOperator)op).setNumThreads(k);
		}
	}

	private static void exec(ExecutionContext ec, Instruction ins){
		final BasicProgramBlock pb = new BasicProgramBlock(null);
		pb.getInstructions().clear();
		pb.getInstructions().add(ins);
		
		try {
			// execute single instruction
			// TODO move this thread naming to Netty thread creation!
			Thread curThread = Thread.currentThread();
			long id = curThread.getId();
			Thread.currentThread().setName("FedExec_"+ id);
			pb.execute(ec);
		}
		catch(Exception ex) {
			// ensure all variables are properly released, even in case
			// of failures because federated workers are stateful servers
			ec.getVariables().releaseAcquiredData();
			throw ex;
		}
	}

	private static void adaptToWorkload(ExecutionContext ec, FederatedWorkloadAnalyzer fan,  long tid, Instruction ins){
		if(fan != null){
			CompletableFuture.runAsync(() -> {
				fan.incrementWorkload(ec, tid, ins);
				fan.compressRun(ec, tid);
			});
		}
	}
	
	private static long getOutputNnz(ExecutionContext ec, Instruction ins) {
		if( ins instanceof ComputationCPInstruction ) {
			Data dat = ec.getVariable(((ComputationCPInstruction)ins).getOutput());
			if( dat instanceof MatrixObject )
				return ((MatrixObject)dat).getNnz();
		}
		return -1L;
	}

	private FederatedResponse execUDF(FederatedRequest request, ExecutionContextMap ecm, EventStageModel eventStage) {
		checkNumParams(request.getNumParams(), 1);
		ExecutionContext ec = ecm.get(request.getTID());

		// get function and input parameters
		try {
			FederatedUDF udf = (FederatedUDF) request.getParam(0);
			LOG.debug(udf);

			eventStage.operation = udf.getClass().getSimpleName();

			Data[] inputs = Arrays.stream(udf.getInputIDs())
				.mapToObj(id -> ec.getVariable(String.valueOf(id)))
				.toArray(Data[]::new);

			// trace lineage
			if(DMLScript.LINEAGE)
				LineageItemUtils.traceFedUDF(ec, udf);

			// reuse or execute user-defined function
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
		catch(FederatedWorkerHandlerException ex) {
			LOG.debug("FederatedWorkerHandler Privacy Constraint " +
				"exception thrown when processing EXEC_UDF request ", ex);
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
			FederatedStatistics.removeDataObjects();
		}
		catch(FederatedWorkerHandlerException ex) {
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

	private boolean shouldTryAsyncCompress(){
		final DMLConfig conf = ConfigurationManager.getDMLConfig();
		return CompressConfig.valueOf(conf.getTextValue(DMLConfig.COMPRESSED_LINALG).toUpperCase()) == CompressConfig.TRUE;
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
				channelFuture.channel().close().sync();
			}
		}
	}
}
