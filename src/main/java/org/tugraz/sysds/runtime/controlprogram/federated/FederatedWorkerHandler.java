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

import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.ChannelInboundHandlerAdapter;
import org.apache.commons.lang.exception.ExceptionUtils;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.wink.json4j.JSONObject;
import org.tugraz.sysds.common.Types;
import org.tugraz.sysds.conf.ConfigurationManager;
import org.tugraz.sysds.parser.DataExpression;
import org.tugraz.sysds.runtime.DMLRuntimeException;
import org.tugraz.sysds.runtime.controlprogram.caching.MatrixObject;
import org.tugraz.sysds.runtime.controlprogram.caching.TensorObject;
import org.tugraz.sysds.runtime.controlprogram.parfor.util.IDSequence;
import org.tugraz.sysds.runtime.instructions.cp.Data;
import org.tugraz.sysds.runtime.instructions.cp.ListObject;
import org.tugraz.sysds.runtime.io.IOUtilFunctions;
import org.tugraz.sysds.runtime.matrix.data.InputInfo;
import org.tugraz.sysds.runtime.matrix.data.OutputInfo;
import org.tugraz.sysds.runtime.meta.MatrixCharacteristics;
import org.tugraz.sysds.runtime.meta.MetaDataFormat;
import org.tugraz.sysds.utils.JSONHelper;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.Map;

import static org.tugraz.sysds.parser.DataExpression.FORMAT_TYPE_VALUE_BINARY;
import static org.tugraz.sysds.parser.DataExpression.FORMAT_TYPE_VALUE_LIBSVM;
import static org.tugraz.sysds.parser.DataExpression.FORMAT_TYPE_VALUE_MATRIXMARKET;
import static org.tugraz.sysds.parser.DataExpression.FORMAT_TYPE_VALUE_TEXT;

public class FederatedWorkerHandler extends ChannelInboundHandlerAdapter {
	private final IDSequence _seq;
	private Map<Long, Data> _vars;
	
	public FederatedWorkerHandler(IDSequence seq, Map<Long, Data> vars) {
		_seq = seq;
		_vars = vars;
	}
	
	@Override
	public void channelRead(ChannelHandlerContext ctx, Object msg) {
		//TODO modify logging to use actual logger not stdout,
		//otherwise the stdout queue might fill up if nobody is reading from it
		
		//TODO could we modularize this method a bit?
		System.out.println("[Federated Worker] Received: " + msg.getClass().getSimpleName());
		FederatedRequest request;
		if( msg instanceof FederatedRequest )
			request = (FederatedRequest) msg;
		else
			throw new DMLRuntimeException("FederatedWorkerHandler: Received object no instance of `FederatedRequest`.");
		FederatedResponse response = null;
		FederatedRequest.FedMethod method = request.getMethod();
		System.out.println("[Federated Worker] Received command: " + method.name());
		synchronized (_seq) {
			switch (method) {
				case READ:
					try {
						// params: filename
						checkNumParams(request.getNumParams(), 1);
						String filename = (String) request.getParam(0);
						MatrixCharacteristics mc = new MatrixCharacteristics();
						mc.setBlocksize(ConfigurationManager.getBlocksize());
						MatrixObject mo = new MatrixObject(Types.ValueType.FP64, filename);
						OutputInfo oi = null;
						InputInfo ii = null;
						// read metadata
						try {
							String mtdname = DataExpression.getMTDFileName(filename);
							Path path = new Path(mtdname);
							try (FileSystem fs = IOUtilFunctions.getFileSystem(mtdname)) {
								if( fs.exists(path) ) {
									try (BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(path)))) {
										JSONObject mtd = JSONHelper.parse(br);
										mc.setRows(mtd.getLong(DataExpression.READROWPARAM));
										mc.setCols(mtd.getLong(DataExpression.READCOLPARAM));
										String format = mtd.getString(DataExpression.FORMAT_TYPE);
										if( format.equalsIgnoreCase(FORMAT_TYPE_VALUE_TEXT) ) {
											oi = OutputInfo.TextCellOutputInfo;
										}
										else if( format.equalsIgnoreCase(FORMAT_TYPE_VALUE_BINARY) ) {
											oi = OutputInfo.BinaryBlockOutputInfo;
										}
										else if( format.equalsIgnoreCase(FORMAT_TYPE_VALUE_MATRIXMARKET) ) {
											oi = OutputInfo.MatrixMarketOutputInfo;
										}
										else if( format.equalsIgnoreCase(FORMAT_TYPE_VALUE_LIBSVM) ) {
											oi = OutputInfo.LIBSVMOutputInfo;
										}
										else {
											response = new FederatedResponse(FederatedResponse.Type.ERROR,
													"Could not figure out correct file format form metadata file");
											break;
										}
										ii = OutputInfo.getMatchingInputInfo(oi);
									}
								}
							}
						}
						catch (Exception ex) {
							throw new DMLRuntimeException(ex);
						}
						MetaDataFormat mdf = new MetaDataFormat(mc, oi, ii);
						mo.setMetaData(mdf);
						
						long id = _seq.getNextID();
						_vars.put(id, mo);
						response = new FederatedResponse(FederatedResponse.Type.SUCCESS, id);
					}
					catch (DMLRuntimeException exception) {
						response = new FederatedResponse(FederatedResponse.Type.ERROR,
								ExceptionUtils.getFullStackTrace(exception));
					}
					finally {
						ctx.writeAndFlush(response);
					}
					break;
				
				case TRANSFER:
					// params: varID
					int numParams = request.getNumParams();
					checkNumParams(numParams, 1);
					long varID = (Long) request.getParam(0);
					Data dataObject = _vars.get(varID);
					switch (dataObject.getDataType()) {
						case TENSOR:
							response = new FederatedResponse(FederatedResponse.Type.SUCCESS,
									((TensorObject) dataObject).acquireReadAndRelease());
							break;
						case MATRIX:
							response = new FederatedResponse(FederatedResponse.Type.SUCCESS,
									((MatrixObject) dataObject).acquireReadAndRelease());
							break;
						case LIST:
							response = new FederatedResponse(FederatedResponse.Type.SUCCESS,
									((ListObject) dataObject).getData());
							break;
						// TODO rest of the possible datatypes
						default:
							response = new FederatedResponse(FederatedResponse.Type.ERROR,
									"FederatedWorkerHandler: Not possible to send datatype " + dataObject.getDataType().name());
					}
					ctx.writeAndFlush(response);
					break;
				case SHUTDOWN:
					System.out.println("[Federated Worker] Shutting down server...");
					ctx.close();
					response = new FederatedResponse(FederatedResponse.Type.SUCCESS_EMPTY);
					break;
				
				default:
					String message = String.format("[Federated Worker] Method %s is not supported.", request.getMethod());
					System.out.println(message);
					response = new FederatedResponse(FederatedResponse.Type.ERROR, message);
			}
			ctx.writeAndFlush(response);
			ctx.close();
		}
	}
	
	private static void checkNumParams(int actual, int... expected) {
		for (int valid : expected) {
			if( actual == valid )
				return;
		}
		throw new DMLRuntimeException("FederatedWorkerHandler: Received wrong amount of params: expected=" + Arrays.toString(expected) + ", actual=" + actual);
	}
	
	@Override
	public void exceptionCaught(ChannelHandlerContext ctx, Throwable cause) {
		cause.printStackTrace();
		ctx.close();
	}
	
}
