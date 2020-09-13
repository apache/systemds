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
import java.util.Arrays;

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
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest.RequestType;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse.ResponseType;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.instructions.InstructionParser;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.instructions.cp.ListObject;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.io.FileFormatPropertiesCSV;
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.meta.MetaDataFormat;
import org.apache.sysds.runtime.privacy.DMLPrivacyException;
import org.apache.sysds.runtime.privacy.PrivacyMonitor;
import org.apache.sysds.runtime.privacy.PrivacyPropagator;
import org.apache.sysds.utils.JSONHelper;
import org.apache.sysds.utils.Statistics;
import org.apache.wink.json4j.JSONObject;

import io.netty.channel.ChannelFuture;
import io.netty.channel.ChannelFutureListener;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.ChannelInboundHandlerAdapter;

public class FederatedWorkerHandler extends ChannelInboundHandlerAdapter {
	protected static Logger log = Logger.getLogger(FederatedWorkerHandler.class);

	private final ExecutionContextMap _ecm;
	
	public FederatedWorkerHandler(ExecutionContextMap ecm) {
		//Note: federated worker handler created for every command;
		//and concurrent parfor threads at coordinator need separate
		//execution contexts at the federated sites too
		_ecm = ecm;
	}

	@Override
	public void channelRead(ChannelHandlerContext ctx, Object msg) {
		if( log.isDebugEnabled() ){
			log.debug("Received: " + msg.getClass().getSimpleName());
		}
		if (!(msg instanceof FederatedRequest[]))
			throw new DMLRuntimeException("FederatedWorkerHandler: Received object no instance of 'FederatedRequest[]'.");
		FederatedRequest[] requests = (FederatedRequest[]) msg;
		FederatedResponse response = null; //last response
		
		for( int i=0; i<requests.length; i++ ) {
			FederatedRequest request = requests[i];
			if( log.isInfoEnabled() ){
				log.info("Executing command " + (i+1) + "/" + requests.length + ": " + request.getType().name());
				if( log.isDebugEnabled() ){
					log.debug("full command: " + request.toString());
				}
			}
			PrivacyMonitor.setCheckPrivacy(request.checkPrivacy());
			PrivacyMonitor.clearCheckedConstraints();
	
			//execute command and handle privacy constraints
			FederatedResponse tmp = executeCommand(request);
			conditionalAddCheckedConstraints(request, tmp);
			
			//select the response for the entire batch of requests
			if (!tmp.isSuccessful()) {
				log.error("Command " + request.getType() + " failed: "
					+ tmp.getErrorMessage() + "full command: \n" + request.toString());
				response = (response == null || response.isSuccessful())
					? tmp : response; //return first error
			}
			else if( request.getType() == RequestType.GET_VAR ) {
				if( response != null && response.isSuccessful() )
					log.error("Multiple GET_VAR are not supported in single batch of requests.");
				response = tmp; //return last get result
			}
			else if( response == null && i == requests.length-1 ) {
				response = tmp; //return last
			}

			if (DMLScript.STATISTICS && request.getType() == RequestType.CLEAR && Statistics.allowWorkerStatistics){
				System.out.println("Federated Worker " + Statistics.display());
				Statistics.reset();
			}
		}
		ctx.writeAndFlush(response).addListener(new CloseListener());
	}

	private static void conditionalAddCheckedConstraints(FederatedRequest request, FederatedResponse response){
		if ( request.checkPrivacy() )
			response.setCheckedConstraints(PrivacyMonitor.getCheckedConstraints());
	}

	private FederatedResponse executeCommand(FederatedRequest request) {
		RequestType method = request.getType();
		try {
			switch (method) {
				case READ_VAR:
					return readData(request); //matrix/frame
				case PUT_VAR:
					return putVariable(request);
				case GET_VAR:
					return getVariable(request);
				case EXEC_INST:
					return execInstruction(request);
				case EXEC_UDF:
					return execUDF(request);
				case CLEAR:
					return execClear();
				default:
					String message = String.format("Method %s is not supported.", method);
					return new FederatedResponse(ResponseType.ERROR,
						new FederatedWorkerHandlerException(message));
			}
		}
		catch (DMLPrivacyException | FederatedWorkerHandlerException ex) {
			return new FederatedResponse(ResponseType.ERROR, ex);
		}
		catch (Exception ex) {
			return new FederatedResponse(ResponseType.ERROR,
				new FederatedWorkerHandlerException("Exception of type "
				+ ex.getClass() + " thrown when processing request", ex));
		}
	}
	
	private FederatedResponse readData(FederatedRequest request) {
		checkNumParams(request.getNumParams(), 2);
		String filename = (String) request.getParam(0);
		DataType dt = DataType.valueOf((String)request.getParam(1));
		return readData(filename, dt, request.getID(), request.getTID());
	}

	private FederatedResponse readData(String filename, Types.DataType dataType, long id, long tid) {
		MatrixCharacteristics mc = new MatrixCharacteristics();
		mc.setBlocksize(ConfigurationManager.getBlocksize());
		CacheableData<?> cd;
		switch (dataType) {
			case MATRIX:
				cd = new MatrixObject(Types.ValueType.FP64, filename);
				break;
			case FRAME:
				cd = new FrameObject(filename);
				break;
			default:
				// should NEVER happen (if we keep request codes in sync with actual behaviour)
				return new FederatedResponse(ResponseType.ERROR,
					new FederatedWorkerHandlerException("Could not recognize datatype"));
		}
		
		// read metadata
		FileFormat fmt = null;
		boolean header = false;
		try {
			String mtdname = DataExpression.getMTDFileName(filename);
			Path path = new Path(mtdname);
			FileSystem fs = IOUtilFunctions.getFileSystem(mtdname); //no auto-close
			try (BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(path)))) {
				JSONObject mtd = JSONHelper.parse(br);
				if (mtd == null)
					return new FederatedResponse(ResponseType.ERROR,
						new FederatedWorkerHandlerException("Could not parse metadata file"));
				mc.setRows(mtd.getLong(DataExpression.READROWPARAM));
				mc.setCols(mtd.getLong(DataExpression.READCOLPARAM));
				if(mtd.containsKey(DataExpression.READNNZPARAM))
					mc.setNonZeros(mtd.getLong(DataExpression.READNNZPARAM));
				if (mtd.has(DataExpression.DELIM_HAS_HEADER_ROW))
					header = mtd.getBoolean(DataExpression.DELIM_HAS_HEADER_ROW);
				cd = (CacheableData<?>) PrivacyPropagator.parseAndSetPrivacyConstraint(cd, mtd);
				fmt = FileFormat.safeValueOf(mtd.getString(DataExpression.FORMAT_TYPE));
			}
		}
		catch (Exception ex) {
			throw new DMLRuntimeException(ex);
		}
		
		//put meta data object in symbol table, read on first operation
		cd.setMetaData(new MetaDataFormat(mc, fmt));
		// TODO send FileFormatProperties with request and use them for CSV, this is currently a workaround so reading
		//  of CSV files works
		cd.setFileFormatProperties(new FileFormatPropertiesCSV(header, DataExpression.DEFAULT_DELIM_DELIMITER,
			DataExpression.DEFAULT_DELIM_SPARSE));
		cd.enableCleanup(false); //guard against deletion
		_ecm.get(tid).setVariable(String.valueOf(id), cd);
		
		if (dataType == Types.DataType.FRAME) {
			FrameObject frameObject = (FrameObject) cd;
			frameObject.acquireRead();
			frameObject.refreshMetaData(); //get block schema
			frameObject.release();
			return new FederatedResponse(ResponseType.SUCCESS,
				new Object[] {id, frameObject.getSchema()});
		}
		return new FederatedResponse(ResponseType.SUCCESS, id);
	}
	
	private FederatedResponse putVariable(FederatedRequest request) {
		checkNumParams(request.getNumParams(), 1);
		String varname = String.valueOf(request.getID());
		ExecutionContext ec = _ecm.get(request.getTID());
		if( ec.containsVariable(varname) ) {
			return new FederatedResponse(ResponseType.ERROR,
				"Variable "+request.getID()+" already existing.");
		}
		
		//wrap transferred cache block into cacheable data
		Data data = null;
		if( request.getParam(0) instanceof CacheBlock )
			data = ExecutionContext.createCacheableData((CacheBlock) request.getParam(0));
		else if( request.getParam(0) instanceof ScalarObject )
			data = (ScalarObject) request.getParam(0);
		
		//set variable and construct empty response
		ec.setVariable(varname, data);
		return new FederatedResponse(ResponseType.SUCCESS_EMPTY);
	}
	
	private FederatedResponse getVariable(FederatedRequest request) {
		checkNumParams(request.getNumParams(), 0);
		ExecutionContext ec = _ecm.get(request.getTID());
		if( !ec.containsVariable(String.valueOf(request.getID())) ) {
			return new FederatedResponse(ResponseType.ERROR,
				"Variable "+request.getID()+" does not exist at federated worker.");
		}
		//get variable and construct response
		Data dataObject = ec.getVariable(String.valueOf(request.getID()));
		dataObject = PrivacyMonitor.handlePrivacy(dataObject);
		switch (dataObject.getDataType()) {
			case TENSOR:
			case MATRIX:
			case FRAME:
				return new FederatedResponse(ResponseType.SUCCESS,
					((CacheableData<?>) dataObject).acquireReadAndRelease());
			case LIST:
				return new FederatedResponse(ResponseType.SUCCESS, ((ListObject) dataObject).getData());
			case SCALAR:
				return new FederatedResponse(ResponseType.SUCCESS, dataObject);
			default:
				return new FederatedResponse(ResponseType.ERROR,
					new FederatedWorkerHandlerException("Unsupported return datatype " + dataObject.getDataType().name()));
		}
	}
	
	private FederatedResponse execInstruction(FederatedRequest request) {
		ExecutionContext ec = _ecm.get(request.getTID());
		BasicProgramBlock pb = new BasicProgramBlock(null);
		pb.getInstructions().clear();
		Instruction receivedInstruction = InstructionParser
			.parseSingleInstruction((String)request.getParam(0));
		pb.getInstructions().add(receivedInstruction);
		try {
			pb.execute(ec); //execute single instruction
		}
		catch(Exception ex) {
			return new FederatedResponse(ResponseType.ERROR, new FederatedWorkerHandlerException(
				"Exception of type " + ex.getClass() + " thrown when processing EXEC_INST request", ex));
		}
		return new FederatedResponse(ResponseType.SUCCESS_EMPTY);
	}
	
	private FederatedResponse execUDF(FederatedRequest request) {
		checkNumParams(request.getNumParams(), 1);
		ExecutionContext ec = _ecm.get(request.getTID());
		
		//get function and input parameters
		FederatedUDF udf = (FederatedUDF) request.getParam(0);
		Data[] inputs = Arrays.stream(udf.getInputIDs())
			.mapToObj(id -> ec.getVariable(String.valueOf(id)))
			.map(PrivacyMonitor::handlePrivacy)
			.toArray(Data[]::new);
		
		//execute user-defined function
		try {
			return udf.execute(ec, inputs);
		}
		catch(Exception ex) {
			return new FederatedResponse(ResponseType.ERROR, new FederatedWorkerHandlerException(
				"Exception of type " + ex.getClass() + " thrown when processing EXEC_UDF request", ex));
		}
	}

	private FederatedResponse execClear() {
		try {
			_ecm.clear();
		}
		catch(Exception ex) {
			return new FederatedResponse(ResponseType.ERROR, new FederatedWorkerHandlerException(
				"Exception of type " + ex.getClass() + " thrown when processing CLEAR request", ex));
		}
		return new FederatedResponse(ResponseType.SUCCESS_EMPTY);
	}
	
	private static void checkNumParams(int actual, int... expected) {
		if (Arrays.stream(expected).anyMatch(x -> x == actual))
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
			if (!channelFuture.isSuccess()){
				log.error("Federated Worker Write failed");
				channelFuture
					.channel()
					.writeAndFlush(
						new FederatedResponse(ResponseType.ERROR,
						new FederatedWorkerHandlerException("Error while sending response.")))
					.channel().close().sync();
			}
			else {
				PrivacyMonitor.clearCheckedConstraints();
				channelFuture.channel().close().sync();
			}
		}
	}
}
