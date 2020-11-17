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

package org.apache.sysds.runtime.instructions.fed;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.controlprogram.caching.FrameObject;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRange;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse.ResponseType;
import org.apache.sysds.runtime.controlprogram.federated.FederatedUDF;
import org.apache.sysds.runtime.controlprogram.federated.FederationMap;
import org.apache.sysds.runtime.controlprogram.federated.FederationUtils;
import org.apache.sysds.runtime.functionobjects.ParameterizedBuiltin;
import org.apache.sysds.runtime.functionobjects.ValueFunction;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.matrix.operators.SimpleOperator;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.meta.MetaDataFormat;
import org.apache.sysds.runtime.transform.decode.Decoder;
import org.apache.sysds.runtime.transform.decode.DecoderFactory;
import org.apache.sysds.runtime.transform.encode.Encoder;
import org.apache.sysds.runtime.transform.encode.EncoderComposite;
import org.apache.sysds.runtime.transform.encode.EncoderFactory;
import org.apache.sysds.runtime.transform.encode.EncoderOmit;

public class ParameterizedBuiltinFEDInstruction extends ComputationFEDInstruction {
	protected final LinkedHashMap<String, String> params;

	protected ParameterizedBuiltinFEDInstruction(Operator op, LinkedHashMap<String, String> paramsMap, CPOperand out,
		String opcode, String istr) {
		super(FEDType.ParameterizedBuiltin, op, null, null, out, opcode, istr);
		params = paramsMap;
	}

	public HashMap<String, String> getParameterMap() {
		return params;
	}

	public String getParam(String key) {
		return getParameterMap().get(key);
	}

	public static LinkedHashMap<String, String> constructParameterMap(String[] params) {
		// process all elements in "params" except first(opcode) and last(output)
		LinkedHashMap<String, String> paramMap = new LinkedHashMap<>();

		// all parameters are of form <name=value>
		String[] parts;
		for(int i = 1; i <= params.length - 2; i++) {
			parts = params[i].split(Lop.NAME_VALUE_SEPARATOR);
			paramMap.put(parts[0], parts[1]);
		}

		return paramMap;
	}

	public static ParameterizedBuiltinFEDInstruction parseInstruction(String str) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		// first part is always the opcode
		String opcode = parts[0];
		// last part is always the output
		CPOperand out = new CPOperand(parts[parts.length - 1]);

		// process remaining parts and build a hash map
		LinkedHashMap<String, String> paramsMap = constructParameterMap(parts);

		// determine the appropriate value function
		if(opcode.equalsIgnoreCase("replace") || opcode.equalsIgnoreCase("rmempty")) {
			ValueFunction func = ParameterizedBuiltin.getParameterizedBuiltinFnObject(opcode);
			return new ParameterizedBuiltinFEDInstruction(new SimpleOperator(func), paramsMap, out, opcode, str);
		}
		else if(opcode.equals("transformapply") || opcode.equals("transformdecode")) {
			return new ParameterizedBuiltinFEDInstruction(null, paramsMap, out, opcode, str);
		}
		else {
			throw new DMLRuntimeException(
				"Unsupported opcode (" + opcode + ") for ParameterizedBuiltinFEDInstruction.");
		}
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		String opcode = getOpcode();
		if(opcode.equalsIgnoreCase("replace")) {
			// similar to unary federated instructions, get federated input
			// execute instruction, and derive federated output matrix
			MatrixObject mo = (MatrixObject) getTarget(ec);
			FederatedRequest fr1 = FederationUtils.callInstruction(instString,
				output,
				new CPOperand[] {getTargetOperand()},
				new long[] {mo.getFedMapping().getID()});
			mo.getFedMapping().execute(getTID(), true, fr1);

			// derive new fed mapping for output
			MatrixObject out = ec.getMatrixObject(output);
			out.getDataCharacteristics().set(mo.getDataCharacteristics());
			out.setFedMapping(mo.getFedMapping().copyWithNewID(fr1.getID()));
		}
		else if(opcode.equals("rmempty"))
			rmempty(ec);
		else if(opcode.equalsIgnoreCase("transformdecode"))
			transformDecode(ec);
		else if(opcode.equalsIgnoreCase("transformapply"))
			transformApply(ec);
		else {
			throw new DMLRuntimeException("Unknown opcode : " + opcode);
		}
	}

	private void rmempty(ExecutionContext ec) {
		String margin = params.get("margin");
		if( !(margin.equals("rows") || margin.equals("cols")) )
			throw new DMLRuntimeException("Unspupported margin identifier '"+margin+"'.");

		MatrixObject mo = (MatrixObject) getTarget(ec);
		MatrixObject select = params.containsKey("select") ? ec.getMatrixObject(params.get("select")) : null;
		MatrixObject out = ec.getMatrixObject(output);

		boolean marginRow = params.get("margin").equals("rows");
		boolean k = ((marginRow && mo.getFedMapping().getType().isColPartitioned()) ||
			(!marginRow && mo.getFedMapping().getType().isRowPartitioned()));

		MatrixBlock s = new MatrixBlock();
		if(select == null && k) {
			List<MatrixBlock> colSums = new ArrayList<>();
			mo.getFedMapping().forEachParallel((range, data) -> {
				try {
					FederatedResponse response = data
						.executeFederatedOperation(new FederatedRequest(FederatedRequest.RequestType.EXEC_UDF, -1,
							new GetVector(data.getVarID(), margin.equals("rows"))))
						.get();

					if(!response.isSuccessful())
						response.throwExceptionFromResponse();
					MatrixBlock vector = (MatrixBlock) response.getData()[0];
					synchronized(colSums) {
						colSums.add(vector);
					}
				}
				catch(Exception e) {
					throw new DMLRuntimeException(e);
				}
				return null;
			});
			// find empty in matrix
			BinaryOperator plus = InstructionUtils.parseBinaryOperator("+");
			BinaryOperator greater = InstructionUtils.parseBinaryOperator(">");
			s = colSums.get(0);
			for(int i = 1; i < colSums.size(); i++)
				s = s.binaryOperationsInPlace(plus, colSums.get(i));
			s = s.binaryOperationsInPlace(greater, new MatrixBlock(s.getNumRows(), s.getNumColumns(), 0.0));
			select = ExecutionContext.createMatrixObject(s);

			long varID = FederationUtils.getNextFedDataID();
			ec.setVariable(String.valueOf(varID), select);
			params.put("select", String.valueOf(varID));
			// construct new string
			String[] oldString = InstructionUtils.getInstructionParts(instString);
			String[] newString = new String[oldString.length+1];
			newString[2] = "select="+varID;
			System.arraycopy(oldString, 0, newString, 0,2);
			System.arraycopy(oldString,2, newString, 3, newString.length-3);
			instString = instString.replace(InstructionUtils.concatOperands(oldString), InstructionUtils.concatOperands(newString));
		}

		if (select == null) {
			FederatedRequest fr1 = FederationUtils.callInstruction(instString, output,
				new CPOperand[] {getTargetOperand()},
				new long[] {mo.getFedMapping().getID()});
			mo.getFedMapping().execute(getTID(), true, fr1);
			out.setFedMapping(mo.getFedMapping().copyWithNewID(fr1.getID()));
		}
		else if (!k) {
			//construct commands: broadcast , fed rmempty, clean broadcast
			FederatedRequest[] fr1 = mo.getFedMapping().broadcastSliced(select, !marginRow);
			FederatedRequest fr2 = FederationUtils.callInstruction(instString,
				output,
				new CPOperand[] {getTargetOperand(), new CPOperand(params.get("select"), ValueType.FP64, DataType.MATRIX)},
				new long[] {mo.getFedMapping().getID(), fr1[0].getID()});
			FederatedRequest fr3 = mo.getFedMapping().cleanup(getTID(), fr1[0].getID());

			//execute federated operations and set output
			mo.getFedMapping().execute(getTID(), true, fr1, fr2, fr3);
			out.setFedMapping(mo.getFedMapping().copyWithNewID(fr2.getID()));
		} else {
			//construct commands: broadcast , fed rmempty, clean broadcast
			FederatedRequest fr1 = mo.getFedMapping().broadcast(select);
			FederatedRequest fr2 = FederationUtils.callInstruction(instString,
				output,
				new CPOperand[] {getTargetOperand(), new CPOperand(params.get("select"), ValueType.FP64, DataType.MATRIX)},
				new long[] {mo.getFedMapping().getID(), fr1.getID()});
			FederatedRequest fr3 = mo.getFedMapping().cleanup(getTID(), fr1.getID());

			//execute federated operations and set output
			mo.getFedMapping().execute(getTID(), true, fr1, fr2, fr3);
			out.setFedMapping(mo.getFedMapping().copyWithNewID(fr2.getID()));
		}

		// new ranges
		Map<FederatedRange, int[]> dcs = new HashMap<>();
		Map<FederatedRange, int[]> finalDcs1 = dcs;
		out.getFedMapping().forEachParallel((range, data) -> {
			try {
				FederatedResponse response = data
					.executeFederatedOperation(new FederatedRequest(FederatedRequest.RequestType.EXEC_UDF, -1,
						new GetDataCharacteristics(data.getVarID())))
					.get();

				if(!response.isSuccessful())
					response.throwExceptionFromResponse();
				int[] subRangeCharacteristics = (int[]) response.getData()[0];
				synchronized(finalDcs1) {
					finalDcs1.put(range, subRangeCharacteristics);
				}
			}
			catch(Exception e) {
				throw new DMLRuntimeException(e);
			}
			return null;
		});
		dcs = finalDcs1;
		out.getDataCharacteristics().set(mo.getDataCharacteristics());
		for(int i = 0; i < mo.getFedMapping().getFederatedRanges().length; i++) {
			int[] newRange = dcs.get(out.getFedMapping().getFederatedRanges()[i]);

			out.getFedMapping().getFederatedRanges()[i].setBeginDim(0,
				(out.getFedMapping().getFederatedRanges()[i].getBeginDims()[0] == 0 ||
					i == 0) ? 0 : out.getFedMapping().getFederatedRanges()[i - 1].getEndDims()[0]);

			out.getFedMapping().getFederatedRanges()[i].setEndDim(0,
				out.getFedMapping().getFederatedRanges()[i].getBeginDims()[0] + newRange[0]);

			out.getFedMapping().getFederatedRanges()[i].setBeginDim(1,
				(out.getFedMapping().getFederatedRanges()[i].getBeginDims()[1] == 0 ||
					i == 0) ? 0 : out.getFedMapping().getFederatedRanges()[i - 1].getEndDims()[1]);

			out.getFedMapping().getFederatedRanges()[i].setEndDim(1,
				out.getFedMapping().getFederatedRanges()[i].getBeginDims()[1] + newRange[1]);
		}

		out.getDataCharacteristics().set(out.getFedMapping().getMaxIndexInRange(0),
			out.getFedMapping().getMaxIndexInRange(1), (int) mo.getBlocksize());
	}

	private void transformDecode(ExecutionContext ec) {
		// acquire locks
		MatrixObject mo = ec.getMatrixObject(params.get("target"));
		FrameBlock meta = ec.getFrameInput(params.get("meta"));
		String spec = params.get("spec");

		Decoder globalDecoder = DecoderFactory
			.createDecoder(spec, meta.getColumnNames(), null, meta, (int) mo.getNumColumns());

		FederationMap fedMapping = mo.getFedMapping();

		ValueType[] schema = new ValueType[(int) mo.getNumColumns()];
		long varID = FederationUtils.getNextFedDataID();
		FederationMap decodedMapping = fedMapping.mapParallel(varID, (range, data) -> {
			long[] beginDims = range.getBeginDims();
			long[] endDims = range.getEndDims();
			int colStartBefore = (int) beginDims[1];

			// update begin end dims (column part) considering columns added by dummycoding
			globalDecoder.updateIndexRanges(beginDims, endDims);

			// get the decoder segment that is relevant for this federated worker
			Decoder decoder = globalDecoder
				.subRangeDecoder((int) beginDims[1] + 1, (int) endDims[1] + 1, colStartBefore);

			FrameBlock metaSlice = new FrameBlock();
			synchronized(meta) {
				meta.slice(0, meta.getNumRows() - 1, (int) beginDims[1], (int) endDims[1] - 1, metaSlice);
			}

			FederatedResponse response;
			try {
				response = data.executeFederatedOperation(new FederatedRequest(FederatedRequest.RequestType.EXEC_UDF,
					-1, new DecodeMatrix(data.getVarID(), varID, metaSlice, decoder))).get();
				if(!response.isSuccessful())
					response.throwExceptionFromResponse();

				ValueType[] subSchema = (ValueType[]) response.getData()[0];
				synchronized(schema) {
					// It would be possible to assert that different federated workers don't give different value
					// types for the same columns, but the performance impact is not worth the effort
					System.arraycopy(subSchema, 0, schema, colStartBefore, subSchema.length);
				}
			}
			catch(Exception e) {
				throw new DMLRuntimeException(e);
			}
			return null;
		});

		// construct a federated matrix with the encoded data
		FrameObject decodedFrame = ec.getFrameObject(output);
		decodedFrame.setSchema(globalDecoder.getSchema());
		decodedFrame.getDataCharacteristics().set(mo.getDataCharacteristics());
		decodedFrame.getDataCharacteristics().setCols(globalDecoder.getSchema().length);
		// set the federated mapping for the matrix
		decodedFrame.setFedMapping(decodedMapping);

		// release locks
		ec.releaseFrameInput(params.get("meta"));
	}

	private void transformApply(ExecutionContext ec) {
		// acquire locks
		FrameObject fo = ec.getFrameObject(params.get("target"));
		FrameBlock meta = ec.getFrameInput(params.get("meta"));
		String spec = params.get("spec");

		FederationMap fedMapping = fo.getFedMapping();

		// get column names for the EncoderFactory
		String[] colNames = new String[(int) fo.getNumColumns()];
		Arrays.fill(colNames, "");

		fedMapping.forEachParallel((range, data) -> {
			try {
				FederatedResponse response = data
					.executeFederatedOperation(new FederatedRequest(FederatedRequest.RequestType.EXEC_UDF, -1,
						new GetColumnNames(data.getVarID())))
					.get();

				// no synchronization necessary since names should anyway match
				String[] subRangeColNames = (String[]) response.getData()[0];
				System.arraycopy(subRangeColNames, 0, colNames, (int) range.getBeginDims()[1], subRangeColNames.length);
			}
			catch(Exception e) {
				throw new DMLRuntimeException(e);
			}
			return null;
		});

		Encoder globalEncoder = EncoderFactory.createEncoder(spec, colNames, colNames.length, meta);

		// check if EncoderOmit exists
		List<Encoder> encoders = ((EncoderComposite) globalEncoder).getEncoders();
		int omitIx = -1;
		for(int i = 0; i < encoders.size(); i++) {
			if(encoders.get(i) instanceof EncoderOmit) {
				omitIx = i;
				break;
			}
		}
		if(omitIx != -1) {
			// extra step, build the omit encoder: we need information about all the rows to omit, if our federated
			// ranges are split up row-wise we need to build the encoder separately and combine it
			buildOmitEncoder(fedMapping, encoders, omitIx);
		}

		MultiReturnParameterizedBuiltinFEDInstruction
			.encodeFederatedFrames(fedMapping, globalEncoder, ec.getMatrixObject(getOutputVariableName()));

		// release locks
		ec.releaseFrameInput(params.get("meta"));
	}

	private static void buildOmitEncoder(FederationMap fedMapping, List<Encoder> encoders, int omitIx) {
		Encoder omitEncoder = encoders.get(omitIx);
		EncoderOmit newOmit = new EncoderOmit(true);
		fedMapping.forEachParallel((range, data) -> {
			try {
				EncoderOmit subRangeEncoder = (EncoderOmit) omitEncoder.subRangeEncoder(range.asIndexRange().add(1));
				FederatedResponse response = data
					.executeFederatedOperation(new FederatedRequest(FederatedRequest.RequestType.EXEC_UDF, -1,
						new InitRowsToRemoveOmit(data.getVarID(), subRangeEncoder)))
					.get();

				// no synchronization necessary since names should anyway match
				Encoder builtEncoder = (Encoder) response.getData()[0];
				newOmit.mergeAt(builtEncoder, (int) (range.getBeginDims()[0] + 1), (int) (range.getBeginDims()[1] + 1));
			}
			catch(Exception e) {
				throw new DMLRuntimeException(e);
			}
			return null;
		});
		encoders.remove(omitIx);
		encoders.add(omitIx, newOmit);
	}

	public CacheableData<?> getTarget(ExecutionContext ec) {
		return ec.getCacheableData(params.get("target"));
	}

	private CPOperand getTargetOperand() {
		return new CPOperand(params.get("target"), ValueType.FP64, DataType.MATRIX);
	}

	public static class DecodeMatrix extends FederatedUDF {
		private static final long serialVersionUID = 2376756757742169692L;
		private final long _outputID;
		private final FrameBlock _meta;
		private final Decoder _decoder;

		public DecodeMatrix(long input, long outputID, FrameBlock meta, Decoder decoder) {
			super(new long[] {input});
			_outputID = outputID;
			_meta = meta;
			_decoder = decoder;
		}

		public FederatedResponse execute(ExecutionContext ec, Data... data) {
			MatrixObject mo = (MatrixObject) data[0];
			MatrixBlock mb = mo.acquireRead();
			String[] colNames = _meta.getColumnNames();

			FrameBlock fbout = _decoder.decode(mb, new FrameBlock(_decoder.getSchema()));
			fbout.setColumnNames(Arrays.copyOfRange(colNames, 0, fbout.getNumColumns()));

			// copy characteristics
			MatrixCharacteristics mc = new MatrixCharacteristics(mo.getDataCharacteristics());
			FrameObject fo = new FrameObject(OptimizerUtils.getUniqueTempFileName(),
				new MetaDataFormat(mc, Types.FileFormat.BINARY));
			// set the encoded data
			fo.acquireModify(fbout);
			fo.release();
			mo.release();

			// add it to the list of variables
			ec.setVariable(String.valueOf(_outputID), fo);
			// return schema
			return new FederatedResponse(ResponseType.SUCCESS, new Object[] {fo.getSchema()});
		}
	}

	private static class GetColumnNames extends FederatedUDF {
		private static final long serialVersionUID = -7831469841164270004L;

		public GetColumnNames(long varID) {
			super(new long[] {varID});
		}

		@Override
		public FederatedResponse execute(ExecutionContext ec, Data... data) {
			FrameBlock fb = ((FrameObject) data[0]).acquireReadAndRelease();
			// return column names
			return new FederatedResponse(ResponseType.SUCCESS, new Object[] {fb.getColumnNames()});
		}
	}

	private static class InitRowsToRemoveOmit extends FederatedUDF {
		private static final long serialVersionUID = -8196730717390438411L;

		EncoderOmit _encoder;

		public InitRowsToRemoveOmit(long varID, EncoderOmit encoder) {
			super(new long[] {varID});
			_encoder = encoder;
		}

		@Override
		public FederatedResponse execute(ExecutionContext ec, Data... data) {
			FrameBlock fb = ((FrameObject) data[0]).acquireReadAndRelease();
			_encoder.build(fb);
			return new FederatedResponse(ResponseType.SUCCESS, new Object[] {_encoder});
		}
	}

	private static class GetDataCharacteristics extends FederatedUDF {

		private static final long serialVersionUID = 578461386177730925L;

		public GetDataCharacteristics(long varID) {
			super(new long[] {varID});
		}

		@Override
		public FederatedResponse execute(ExecutionContext ec, Data... data) {
			MatrixBlock mb = ((MatrixObject) data[0]).acquireReadAndRelease();
			int r = mb.getDenseBlockValues() != null ? mb.getNumRows() : 0;
			int c = mb.getDenseBlockValues() != null ? mb.getNumColumns(): 0;
			return new FederatedResponse(ResponseType.SUCCESS, new int[] {r, c});
		}
	}

	private static class GetVector extends FederatedUDF {

		private static final long serialVersionUID = -1003061862215703768L;
		private final boolean _marginRow;

		public GetVector(long varID, boolean marginRow) {
			super(new long[] {varID});
			_marginRow = marginRow;
		}

		@Override
		public FederatedResponse execute(ExecutionContext ec, Data... data) {
			MatrixBlock mb = ((MatrixObject) data[0]).acquireReadAndRelease();

			BinaryOperator plus = InstructionUtils.parseBinaryOperator("+");
			BinaryOperator greater = InstructionUtils.parseBinaryOperator(">");
			int len = _marginRow ? mb.getNumColumns() : mb.getNumRows();
			MatrixBlock tmp1 = _marginRow ? mb.slice(0, mb.getNumRows() - 1, 0, 0, new MatrixBlock()) : mb
				.slice(0, 0, 0, mb.getNumColumns() - 1, new MatrixBlock());
			for(int i = 1; i < len; i++) {
				MatrixBlock tmp2 = _marginRow ? mb.slice(0, mb.getNumRows() - 1, i, i, new MatrixBlock()) : mb
					.slice(i, i, 0, mb.getNumColumns() - 1, new MatrixBlock());
				tmp1 = tmp1.binaryOperationsInPlace(plus, tmp2);
			}
			tmp1 = tmp1.binaryOperationsInPlace(greater, new MatrixBlock(tmp1.getNumRows(), tmp1.getNumColumns(), 0.0));
			return new FederatedResponse(ResponseType.SUCCESS, tmp1);
		}
	}
}
