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

import io.netty.channel.ChannelFuture;
import io.netty.channel.ChannelFutureListener;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.ChannelInboundHandlerAdapter;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.log4j.Logger;
import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.parser.DataExpression;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.controlprogram.caching.FrameObject;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.caching.TensorObject;
import org.apache.sysds.runtime.controlprogram.parfor.util.IDSequence;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.instructions.cp.ListObject;
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.runtime.matrix.data.LibMatrixAgg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.AggregateBinaryOperator;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.meta.MetaDataFormat;
import org.apache.sysds.runtime.privacy.DMLPrivacyException;
import org.apache.sysds.runtime.privacy.PrivacyMonitor;
import org.apache.sysds.runtime.privacy.PrivacyPropagator;
import org.apache.sysds.utils.JSONHelper;
import org.apache.wink.json4j.JSONObject;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.Map;

public class FederatedWorkerHandler extends ChannelInboundHandlerAdapter {
	protected static Logger log = Logger.getLogger(FederatedWorkerHandler.class);

	private final IDSequence _seq;
	private Map<Long, Data> _vars;

	public FederatedWorkerHandler(IDSequence seq, Map<Long, Data> _vars2) {
		_seq = seq;
		_vars = _vars2;
	}

	@Override
	public void channelRead(ChannelHandlerContext ctx, Object msg) {
		log.debug("Received: " + msg.getClass().getSimpleName());
		FederatedRequest request;
		if (msg instanceof FederatedRequest)
			request = (FederatedRequest) msg;
		else
			throw new DMLRuntimeException("FederatedWorkerHandler: Received object no instance of `FederatedRequest`.");
		FederatedRequest.FedMethod method = request.getMethod();
		log.debug("Received command: " + method.name());
		PrivacyMonitor.setCheckPrivacy(request.checkPrivacy());
		PrivacyMonitor.clearCheckedConstraints();

		synchronized (_seq) {
			FederatedResponse response = constructResponse(request);
			conditionalAddCheckedConstraints(request, response);
			if (!response.isSuccessful())
				log.error("Method " + method + " failed: " + response.getErrorMessage());
			ctx.writeAndFlush(response).addListener(new CloseListener());
		}
	}

	private void conditionalAddCheckedConstraints(FederatedRequest request, FederatedResponse response){
		if ( request.checkPrivacy() )
			response.setCheckedConstraints(PrivacyMonitor.getCheckedConstraints());
	}

	private FederatedResponse constructResponse(FederatedRequest request) {
		FederatedRequest.FedMethod method = request.getMethod();
		try {
			switch (method) {
				case READ_MATRIX:
					return readData(request, Types.DataType.MATRIX);
				case READ_FRAME:
					return readData(request, Types.DataType.FRAME);
				case MATVECMULT:
					return executeMatVecMult(request);
				case TRANSFER:
					return getVariableData(request);
				case AGGREGATE:
					return executeAggregation(request);
				case SCALAR:
					return executeScalarOperation(request);
				default:
					String message = String.format("Method %s is not supported.", method);
					return new FederatedResponse(FederatedResponse.Type.ERROR, new FederatedWorkerHandlerException(message));
			}
		}
		catch (DMLPrivacyException | FederatedWorkerHandlerException exception) {
			return new FederatedResponse(FederatedResponse.Type.ERROR, exception);
		}
		catch (Exception exception) {
			return new FederatedResponse(FederatedResponse.Type.ERROR, 
				new FederatedWorkerHandlerException("Exception of type " 
				+ exception.getClass() + " thrown when processing request"));
		}
	}

	private FederatedResponse readData(FederatedRequest request, Types.DataType dataType) {
		checkNumParams(request.getNumParams(), 1);
		String filename = (String) request.getParam(0);
		return readData(filename, dataType);
	}

	private FederatedResponse readData(String filename, Types.DataType dataType) {
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
				return new FederatedResponse(FederatedResponse.Type.ERROR, 
					new FederatedWorkerHandlerException("Could not recognize datatype"));
		}
		
		// read metadata
		FileFormat fmt = null;
		try {
			String mtdname = DataExpression.getMTDFileName(filename);
			Path path = new Path(mtdname);
			try (FileSystem fs = IOUtilFunctions.getFileSystem(mtdname)) {
				try (BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(path)))) {
					JSONObject mtd = JSONHelper.parse(br);
					if (mtd == null)
						return new FederatedResponse(FederatedResponse.Type.ERROR, new FederatedWorkerHandlerException("Could not parse metadata file"));
					mc.setRows(mtd.getLong(DataExpression.READROWPARAM));
					mc.setCols(mtd.getLong(DataExpression.READCOLPARAM));
					cd = PrivacyPropagator.parseAndSetPrivacyConstraint(cd, mtd);
					fmt = FileFormat.safeValueOf(mtd.getString(DataExpression.FORMAT_TYPE));
				}
			}
		}
		catch (Exception ex) {
			throw new DMLRuntimeException(ex);
		}
		cd.setMetaData(new MetaDataFormat(mc, fmt));
		cd.acquireRead();
		cd.refreshMetaData();
		cd.release();

		long id = _seq.getNextID();
		_vars.put(id, cd);
		if (dataType == Types.DataType.FRAME) {
			FrameObject frameObject = (FrameObject) cd;
			return new FederatedResponse(FederatedResponse.Type.SUCCESS, new Object[] {id, frameObject.getSchema()});
		}
		return new FederatedResponse(FederatedResponse.Type.SUCCESS, id);
	}

	private FederatedResponse executeMatVecMult(FederatedRequest request) {
		checkNumParams(request.getNumParams(), 3);
		MatrixBlock vector = (MatrixBlock) request.getParam(0);
		boolean isMatVecMult = (Boolean) request.getParam(1);
		long varID = (Long) request.getParam(2);

		return executeMatVecMult(varID, vector, isMatVecMult);
	}

	private FederatedResponse executeMatVecMult(long varID, MatrixBlock vector, boolean isMatVecMult) {
		MatrixObject matTo = (MatrixObject) _vars.get(varID);
		matTo = PrivacyMonitor.handlePrivacy(matTo);
		MatrixBlock matBlock1 = matTo.acquireReadAndRelease();
		// TODO other datatypes
		AggregateBinaryOperator ab_op = InstructionUtils
			.getMatMultOperator(OptimizerUtils.getConstrainedNumThreads(0));
		MatrixBlock result = isMatVecMult ?
			matBlock1.aggregateBinaryOperations(matBlock1, vector, new MatrixBlock(), ab_op) :
			vector.aggregateBinaryOperations(vector, matBlock1, new MatrixBlock(), ab_op);
		return new FederatedResponse(FederatedResponse.Type.SUCCESS, result);
	}

	private FederatedResponse getVariableData(FederatedRequest request) {
		checkNumParams(request.getNumParams(), 1);
		long varID = (Long) request.getParam(0);
		return getVariableData(varID);
	}

	private FederatedResponse getVariableData(long varID) {
		Data dataObject = _vars.get(varID);
		dataObject = PrivacyMonitor.handlePrivacy(dataObject);
		switch (dataObject.getDataType()) {
			case TENSOR:
				return new FederatedResponse(FederatedResponse.Type.SUCCESS,
					((TensorObject) dataObject).acquireReadAndRelease());
			case MATRIX:
				return new FederatedResponse(FederatedResponse.Type.SUCCESS,
					((MatrixObject) dataObject).acquireReadAndRelease());
			case FRAME:
				return new FederatedResponse(FederatedResponse.Type.SUCCESS,
						((FrameObject) dataObject).acquireReadAndRelease());
			case LIST:
				return new FederatedResponse(FederatedResponse.Type.SUCCESS, ((ListObject) dataObject).getData());
			// TODO rest of the possible datatypes
			default:
				return new FederatedResponse(FederatedResponse.Type.ERROR,
					new FederatedWorkerHandlerException("Not possible to send datatype " + dataObject.getDataType().name()));
		}
	}

	private FederatedResponse executeAggregation(FederatedRequest request) {
		checkNumParams(request.getNumParams(), 2);
		AggregateUnaryOperator operator = (AggregateUnaryOperator) request.getParam(0);
		long varID = (Long) request.getParam(1);
		return executeAggregation(varID, operator);
	}

	private FederatedResponse executeAggregation(long varID, AggregateUnaryOperator operator) {
		Data dataObject = _vars.get(varID);
		if (dataObject.getDataType() != Types.DataType.MATRIX) {
			return new FederatedResponse(FederatedResponse.Type.ERROR, 
				new FederatedWorkerHandlerException("Aggregation only supported for matrices, not for "
				+ dataObject.getDataType().name()));
		}
		MatrixObject matrixObject = (MatrixObject) dataObject;
		matrixObject = PrivacyMonitor.handlePrivacy(matrixObject);
		MatrixBlock matrixBlock = matrixObject.acquireRead();
		// create matrix for calculation with correction
		MatrixCharacteristics mc = new MatrixCharacteristics();
		// find out the characteristics after aggregation
		operator.indexFn.computeDimension(matrixObject.getDataCharacteristics(), mc);
		// make outBlock right size
		int outNumRows = (int) mc.getRows();
		int outNumCols = (int) mc.getCols();
		if (operator.aggOp.existsCorrection()) {
			// add rows for correction
			int numMissing = operator.aggOp.correction.getNumRemovedRowsColumns();
			if (operator.aggOp.correction.isRows())
				outNumRows += numMissing;
			else
				outNumCols += numMissing;
		}
		MatrixBlock ret = new MatrixBlock(outNumRows, outNumCols, operator.aggOp.initialValue);
		LibMatrixAgg.aggregateUnaryMatrix(matrixBlock, ret, operator);
		// result block without correction
		ret.dropLastRowsOrColumns(operator.aggOp.correction);
		return new FederatedResponse(FederatedResponse.Type.SUCCESS, ret);
	}

	private FederatedResponse executeScalarOperation(FederatedRequest request) {
		checkNumParams(request.getNumParams(), 2);
		ScalarOperator operator = (ScalarOperator) request.getParam(0);
		long varID = (Long) request.getParam(1);
		return executeScalarOperation(varID, operator);
	}

	private FederatedResponse executeScalarOperation(long varID, ScalarOperator operator) {
		Data dataObject = _vars.get(varID);
		dataObject = PrivacyMonitor.handlePrivacy(dataObject);
		if (dataObject.getDataType() != Types.DataType.MATRIX) {
			return new FederatedResponse(FederatedResponse.Type.ERROR,
				new FederatedWorkerHandlerException("FederatedWorkerHandler: ScalarOperator dont support "
					+ dataObject.getDataType().name()));
		}

		MatrixObject matrixObject = (MatrixObject) dataObject;
		MatrixBlock inBlock = matrixObject.acquireRead();
		MatrixBlock retBlock = inBlock.scalarOperations(operator, new MatrixBlock());
		return new FederatedResponse(FederatedResponse.Type.SUCCESS, retBlock);
	}

	@SuppressWarnings("unused")
	private FederatedResponse createMatrixObject(MatrixBlock result) {
		MatrixObject resTo = new MatrixObject(Types.ValueType.FP64, OptimizerUtils.getUniqueTempFileName());
		MetaDataFormat metadata = new MetaDataFormat(
			new MatrixCharacteristics(result.getNumRows(), result.getNumColumns()), FileFormat.BINARY);
		resTo.setMetaData(metadata);
		resTo.acquireModify(result);
		resTo.release();
		long result_var = _seq.getNextID();
		_vars.put(result_var, resTo);
		return new FederatedResponse(FederatedResponse.Type.SUCCESS, result_var);
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
		public void operationComplete(ChannelFuture channelFuture) throws InterruptedException, DMLRuntimeException {
			if (!channelFuture.isSuccess())
				throw new DMLRuntimeException("Federated Worker Write failed");
			PrivacyMonitor.clearCheckedConstraints();
			channelFuture.channel().close().sync();
		}
	}
}
