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

import io.netty.channel.ChannelFuture;
import io.netty.channel.ChannelFutureListener;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.ChannelInboundHandlerAdapter;
import org.apache.commons.lang.exception.ExceptionUtils;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.log4j.Logger;
import org.apache.wink.json4j.JSONObject;
import org.tugraz.sysds.common.Types;
import org.tugraz.sysds.conf.ConfigurationManager;
import org.tugraz.sysds.hops.OptimizerUtils;
import org.tugraz.sysds.parser.DataExpression;
import org.tugraz.sysds.runtime.DMLRuntimeException;
import org.tugraz.sysds.runtime.controlprogram.caching.CacheableData;
import org.tugraz.sysds.runtime.controlprogram.caching.MatrixObject;
import org.tugraz.sysds.runtime.controlprogram.caching.TensorObject;
import org.tugraz.sysds.runtime.controlprogram.parfor.util.IDSequence;
import org.tugraz.sysds.runtime.functionobjects.Multiply;
import org.tugraz.sysds.runtime.functionobjects.Plus;
import org.tugraz.sysds.runtime.instructions.cp.Data;
import org.tugraz.sysds.runtime.instructions.cp.ListObject;
import org.tugraz.sysds.runtime.io.IOUtilFunctions;
import org.tugraz.sysds.runtime.matrix.data.InputInfo;
import org.tugraz.sysds.runtime.matrix.data.LibMatrixAgg;
import org.tugraz.sysds.runtime.matrix.data.MatrixBlock;
import org.tugraz.sysds.runtime.matrix.data.OutputInfo;
import org.tugraz.sysds.runtime.matrix.operators.AggregateBinaryOperator;
import org.tugraz.sysds.runtime.matrix.operators.AggregateOperator;
import org.tugraz.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.tugraz.sysds.runtime.meta.MatrixCharacteristics;
import org.tugraz.sysds.runtime.meta.MetaDataFormat;
import org.tugraz.sysds.utils.JSONHelper;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.Map;

import static org.tugraz.sysds.parser.DataExpression.FORMAT_TYPE_VALUE_BINARY;
import static org.tugraz.sysds.parser.DataExpression.FORMAT_TYPE_VALUE_CSV;
import static org.tugraz.sysds.parser.DataExpression.FORMAT_TYPE_VALUE_LIBSVM;
import static org.tugraz.sysds.parser.DataExpression.FORMAT_TYPE_VALUE_MATRIXMARKET;
import static org.tugraz.sysds.parser.DataExpression.FORMAT_TYPE_VALUE_TEXT;

public class FederatedWorkerHandler extends ChannelInboundHandlerAdapter {
	protected static Logger log = Logger.getLogger(FederatedWorkerHandler.class);
	
	private final IDSequence _seq;
	private Map<Long, CacheableData<?>> _vars;
	
	public FederatedWorkerHandler(IDSequence seq, Map<Long, CacheableData<?>> _vars2) {
		_seq = seq;
		_vars = _vars2;
	}
	
	@Override
	public void channelRead(ChannelHandlerContext ctx, Object msg) {
		//TODO could we modularize this method a bit?
		log.debug("[Federated Worker] Received: " + msg.getClass().getSimpleName());
		FederatedRequest request;
		if (msg instanceof FederatedRequest)
			request = (FederatedRequest) msg;
		else
			throw new DMLRuntimeException("FederatedWorkerHandler: Received object no instance of `FederatedRequest`.");
		FederatedResponse response;
		FederatedRequest.FedMethod method = request.getMethod();
		log.debug("[Federated Worker] Received command: " + method.name());
		synchronized (_seq) {
			switch (method) {
				case READ:
					try {
						// params: filename
						checkNumParams(request.getNumParams(), 1);
						String filename = (String) request.getParam(0);
						response = readMatrix(filename);
					}
					catch (DMLRuntimeException exception) {
						response = new FederatedResponse(FederatedResponse.Type.ERROR,
							ExceptionUtils.getFullStackTrace(exception));
					}
					break;
				
				case MATVECMULT:
					try {
						// params: vector, isMatVecMult, varID
						int numParams = request.getNumParams();
						checkNumParams(numParams, 3);
						MatrixBlock vector = (MatrixBlock) request.getParam(0);
						boolean isMatVecMult = (Boolean) request.getParam(1);
						long varID = (Long) request.getParam(2);
						
						response = executeMatVecMult(varID, vector, isMatVecMult);
					}
					catch (Exception exception) {
						response = new FederatedResponse(FederatedResponse.Type.ERROR,
							ExceptionUtils.getFullStackTrace(exception));
					}
					break;
				
				case TRANSFER:
					// params: varID
					int numParams = request.getNumParams();
					checkNumParams(numParams, 1);
					long varID = (Long) request.getParam(0);
					response = getVariableData(varID);
					break;
				
				case AGGREGATE:
					// params: operatore, varID
					numParams = request.getNumParams();
					checkNumParams(numParams, 2);
					AggregateUnaryOperator operator = (AggregateUnaryOperator) request.getParam(0);
					varID = (Long) request.getParam(1);
					response = executeAggregation(varID, operator);
					break;
				
				default:
					String message = String.format(
						"[Federated Worker] Method %s is not supported.", request.getMethod());
					response = new FederatedResponse(FederatedResponse.Type.ERROR, message);
			}
			if (!response.isSuccessful())
				log.error("[Federated Worker] Method " + request.getMethod() + " failed: " + response.getErrorMessage());
			ctx.writeAndFlush(response).addListener(new CloseListener());
		}
	}
	
	private FederatedResponse executeAggregation(long varID, AggregateUnaryOperator operator) {
		Data dataObject = _vars.get(varID);
		if (dataObject.getDataType() != Types.DataType.MATRIX) {
			return new FederatedResponse(FederatedResponse.Type.ERROR,
				"FederatedWorkerHandler: Aggregation only supported for matrices, not for " +
					dataObject.getDataType().name());
		}
		MatrixObject matrixObject = (MatrixObject) dataObject;
		MatrixBlock matrixBlock = matrixObject.acquireRead();
		// create matrix for calculation with correction
		MatrixCharacteristics mc = new MatrixCharacteristics();
		// find out the characteristics after aggregation
		operator.indexFn.computeDimension(matrixObject.getDataCharacteristics(), mc);
		// make outBlock right size
		int outNumRows = (int) mc.getRows();
		int outNumCols = (int) mc.getCols();
		if (operator.aggOp.correctionExists) {
			// add rows for correction
			int numMissing = operator.aggOp.correctionLocation.getNumRemovedRowsColumns();
			if (operator.aggOp.correctionLocation.isRows())
				outNumRows += numMissing;
			else
				outNumCols += numMissing;
		}
		MatrixBlock ret = new MatrixBlock(outNumRows, outNumCols, operator.aggOp.initialValue);
		try {
			LibMatrixAgg.aggregateUnaryMatrix(matrixBlock, ret, operator);
		}
		catch (Exception e) {
			return new FederatedResponse(FederatedResponse.Type.ERROR, "FederatedWorkerHandler: " + e);
		}
		// result block without correction
		ret.dropLastRowsOrColumns(operator.aggOp.correctionLocation);
		return new FederatedResponse(FederatedResponse.Type.SUCCESS, ret);
	}
	
	private FederatedResponse getVariableData(long varID) {
		FederatedResponse response;
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
				response = new FederatedResponse(FederatedResponse.Type.SUCCESS, ((ListObject) dataObject).getData());
				break;
			// TODO rest of the possible datatypes
			default:
				response = new FederatedResponse(FederatedResponse.Type.ERROR,
					"FederatedWorkerHandler: Not possible to send datatype " + dataObject.getDataType().name());
		}
		return response;
	}
	
	private FederatedResponse executeMatVecMult(long varID, MatrixBlock vector, boolean isMatVecMult) {
		MatrixObject matTo = (MatrixObject) _vars.get(varID);
		MatrixBlock matBlock1 = matTo.acquireReadAndRelease();
		// TODO other datatypes
		AggregateBinaryOperator ab_op = new AggregateBinaryOperator(
			Multiply.getMultiplyFnObject(), new AggregateOperator(0, Plus.getPlusFnObject()));
		MatrixBlock result = isMatVecMult ?
			matBlock1.aggregateBinaryOperations(matBlock1, vector, new MatrixBlock(), ab_op) :
			vector.aggregateBinaryOperations(vector, matBlock1, new MatrixBlock(), ab_op);
		return new FederatedResponse(FederatedResponse.Type.SUCCESS, result);
	}
	
	private FederatedResponse readMatrix(String filename) {
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
				try (BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(path)))) {
					JSONObject mtd = JSONHelper.parse(br);
					if (mtd == null)
						return new FederatedResponse(FederatedResponse.Type.ERROR, "Could not parse metadata file");
					mc.setRows(mtd.getLong(DataExpression.READROWPARAM));
					mc.setCols(mtd.getLong(DataExpression.READCOLPARAM));
					String format = mtd.getString(DataExpression.FORMAT_TYPE);
					if (format.equalsIgnoreCase(FORMAT_TYPE_VALUE_TEXT)) {
						oi = OutputInfo.TextCellOutputInfo;
					}
					else if (format.equalsIgnoreCase(FORMAT_TYPE_VALUE_BINARY)) {
						oi = OutputInfo.BinaryBlockOutputInfo;
					}
					else if (format.equalsIgnoreCase(FORMAT_TYPE_VALUE_MATRIXMARKET)) {
						oi = OutputInfo.MatrixMarketOutputInfo;
					}
					else if (format.equalsIgnoreCase(FORMAT_TYPE_VALUE_LIBSVM)) {
						oi = OutputInfo.LIBSVMOutputInfo;
					}
					else if (format.equalsIgnoreCase(FORMAT_TYPE_VALUE_CSV)) {
						oi = OutputInfo.CSVOutputInfo;
					}
					else {
						return new FederatedResponse(FederatedResponse.Type.ERROR,
								"Could not figure out correct file format from metadata file");
					}
					ii = OutputInfo.getMatchingInputInfo(oi);
				}
			}
		}
		catch (Exception ex) {
			throw new DMLRuntimeException(ex);
		}
		MetaDataFormat mdf = new MetaDataFormat(mc, oi, ii);
		mo.setMetaData(mdf);
		mo.acquireRead();
		mo.refreshMetaData();
		mo.release();
		
		long id = _seq.getNextID();
		_vars.put(id, mo);
		return new FederatedResponse(FederatedResponse.Type.SUCCESS, id);
	}
	
	@SuppressWarnings("unused")
	private FederatedResponse createMatrixObject(MatrixBlock result) {
		MatrixObject resTo = new MatrixObject(Types.ValueType.FP64, OptimizerUtils.getUniqueTempFileName());
		MetaDataFormat metadata = new MetaDataFormat(new MatrixCharacteristics(result.getNumRows(),
			result.getNumColumns()), OutputInfo.BinaryBlockOutputInfo, InputInfo.BinaryBlockInputInfo);
		resTo.setMetaData(metadata);
		resTo.acquireModify(result);
		resTo.release();
		long result_var = _seq.getNextID();
		_vars.put(result_var, resTo);
		return new FederatedResponse(FederatedResponse.Type.SUCCESS, result_var);
	}
	
	private static void checkNumParams(int actual, int... expected) {
		if( Arrays.stream(expected).anyMatch(x -> x==actual) )
			return;
		throw new DMLRuntimeException(
			"FederatedWorkerHandler: Received wrong amount of params:" + " expected="
				+ Arrays.toString(expected) + ", actual=" + actual);
	}
	
	@Override
	public void exceptionCaught(ChannelHandlerContext ctx, Throwable cause) {
		cause.printStackTrace();
		ctx.close();
	}
	
	private static class CloseListener implements ChannelFutureListener {
		@Override
		public void operationComplete(ChannelFuture channelFuture) throws Exception {
			if (!channelFuture.isSuccess())
				throw new DMLRuntimeException("[Federated Worker] Write failed");
			channelFuture.channel().close().sync();
		}
	}
}
