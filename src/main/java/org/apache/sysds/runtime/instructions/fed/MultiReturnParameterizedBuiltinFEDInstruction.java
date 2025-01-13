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
import java.util.List;
import java.util.Map;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.LongAdder;
import java.util.stream.Stream;
import java.util.zip.Adler32;
import java.util.zip.Checksum;

import org.apache.commons.lang3.SerializationUtils;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.hops.fedplanner.FTypes;
import org.apache.sysds.hops.fedplanner.FTypes.FType;
import org.apache.sysds.lops.PickByCount;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.controlprogram.caching.FrameObject;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest.RequestType;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse.ResponseType;
import org.apache.sysds.runtime.controlprogram.federated.FederatedUDF;
import org.apache.sysds.runtime.controlprogram.federated.FederationMap;
import org.apache.sysds.runtime.controlprogram.federated.FederationUtils;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.instructions.cp.MultiReturnParameterizedBuiltinCPInstruction;
import org.apache.sysds.runtime.instructions.spark.MultiReturnParameterizedBuiltinSPInstruction;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.lineage.LineageItemUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.transform.encode.ColumnEncoderBin;
import org.apache.sysds.runtime.transform.encode.ColumnEncoderComposite;
import org.apache.sysds.runtime.transform.encode.ColumnEncoderRecode;
import org.apache.sysds.runtime.transform.encode.Encoder;
import org.apache.sysds.runtime.transform.encode.EncoderFactory;
import org.apache.sysds.runtime.transform.encode.MultiColumnEncoder;
import org.apache.sysds.runtime.util.IndexRange;

public class MultiReturnParameterizedBuiltinFEDInstruction extends ComputationFEDInstruction {
	protected final List<CPOperand> _outputs;
	protected final boolean _metaReturn;
	
	private MultiReturnParameterizedBuiltinFEDInstruction(Operator op, CPOperand input1, CPOperand input2,
		List<CPOperand> outputs, boolean metaReturn, String opcode, String istr) {
		super(FEDType.MultiReturnParameterizedBuiltin, op, input1, input2, null, opcode, istr);
		_metaReturn = metaReturn;
		_outputs = outputs;
	}

	public CPOperand getOutput(int i) {
		return _outputs.get(i);
	}

	public static MultiReturnParameterizedBuiltinFEDInstruction parseInstruction(
		MultiReturnParameterizedBuiltinCPInstruction inst, ExecutionContext ec) {
		if(inst.getOpcode().equals("transformencode") && inst.input1.isFrame()) {
			CacheableData<?> fo = ec.getCacheableData(inst.input1);
			if(fo.isFederatedExcept(FType.BROADCAST))
				return MultiReturnParameterizedBuiltinFEDInstruction.parseInstruction(inst);
		}
		return null;
	}

	public static MultiReturnParameterizedBuiltinFEDInstruction parseInstruction(
		MultiReturnParameterizedBuiltinSPInstruction inst, ExecutionContext ec) {
		if(inst.getOpcode().equals("transformencode") && inst.input1.isFrame()) {
			CacheableData<?> fo = ec.getCacheableData(inst.input1);
			if(fo.isFederatedExcept(FType.BROADCAST))
				return MultiReturnParameterizedBuiltinFEDInstruction.parseInstruction(inst);
		}
		return null;
	}

	private static MultiReturnParameterizedBuiltinFEDInstruction parseInstruction(
		MultiReturnParameterizedBuiltinCPInstruction instr) {
		return new MultiReturnParameterizedBuiltinFEDInstruction(instr.getOperator(), instr.input1, instr.input2,
			instr.getOutputs(), instr.getMetaReturn(), instr.getOpcode(), instr.getInstructionString());
	}

	private static MultiReturnParameterizedBuiltinFEDInstruction parseInstruction(
		MultiReturnParameterizedBuiltinSPInstruction instr) {
		return new MultiReturnParameterizedBuiltinFEDInstruction(instr.getOperator(), instr.input1, instr.input2,
			instr.getOutputs(), instr.getMetaReturn(), instr.getOpcode(), instr.getInstructionString());
	}

	public static MultiReturnParameterizedBuiltinFEDInstruction parseInstruction(String str) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		ArrayList<CPOperand> outputs = new ArrayList<>();
		String opcode = parts[0];

		if(opcode.equalsIgnoreCase("transformencode")) {
			// one input and two outputs
			CPOperand in1 = new CPOperand(parts[1]);
			CPOperand in2 = new CPOperand(parts[2]);
			int pos = 3;
			boolean metaReturn = true;
			if( parts.length == 7 ) //no need for meta data
				metaReturn = new CPOperand(parts[pos++]).getLiteral().getBooleanValue();
			outputs.add(new CPOperand(parts[pos], Types.ValueType.FP64, Types.DataType.MATRIX));
			outputs.add(new CPOperand(parts[pos+1], Types.ValueType.STRING, Types.DataType.FRAME));
			return new MultiReturnParameterizedBuiltinFEDInstruction(
				null, in1, in2, outputs, metaReturn, opcode, str);
		}
		else {
			throw new DMLRuntimeException("Invalid opcode in MultiReturnBuiltin instruction: " + opcode);
		}

	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		// obtain and pin input frame
		FrameObject fin = ec.getFrameObject(input1.getName());
		String spec = ec.getScalarInput(input2).getStringValue();

		String[] colNames = new String[(int) fin.getNumColumns()];
		Arrays.fill(colNames, "");

		// the encoder in which the complete encoding information will be aggregated
		MultiColumnEncoder globalEncoder = new MultiColumnEncoder(new ArrayList<>());
		FederationMap fedMapping = fin.getFedMapping();

		boolean containsEquiWidthEncoder = !fin.isFederated(FTypes.FType.ROW) && spec.toLowerCase().contains("equi-height");
		if(containsEquiWidthEncoder) {
			EncoderColnames ret = createGlobalEncoderWithEquiHeight(ec, fin, spec);
			globalEncoder = ret._encoder;
			colNames = ret._colnames;
		} else {
			// first create encoders at the federated workers, then collect them and aggregate them to a single large
			// encoder
			MultiColumnEncoder finalGlobalEncoder = globalEncoder;
			String[] finalColNames = colNames;
			fedMapping.forEachParallel((range, data) -> {
				int columnOffset = (int) range.getBeginDims()[1];

				// create an encoder with the given spec. The columnOffset (which is 0 based) has to be used to
				// tell the federated worker how much the indexes in the spec have to be offset.
				Future<FederatedResponse> responseFuture = data.executeFederatedOperation(new FederatedRequest(
					RequestType.EXEC_UDF,
					-1,
					new CreateFrameEncoder(data.getVarID(), spec, columnOffset + 1)));
				// collect responses with encoders
				try {
					FederatedResponse response = responseFuture.get();
					MultiColumnEncoder encoder = (MultiColumnEncoder) response.getData()[0];

					// merge this encoder into a composite encoder
					synchronized(finalGlobalEncoder) {
						finalGlobalEncoder.mergeAt(encoder, columnOffset, (int) (range.getBeginDims()[0] + 1));
					}
					// no synchronization necessary since names should anyway match
					String[] subRangeColNames = (String[]) response.getData()[1];
					System.arraycopy(subRangeColNames, 0, finalColNames, (int) range.getBeginDims()[1], subRangeColNames.length);
				}
				catch(Exception e) {
					throw new DMLRuntimeException("Federated encoder creation failed: ", e);
				}
				return null;
			});
			globalEncoder = finalGlobalEncoder;
			colNames = finalColNames;
		}

		// sort for consistent encoding in local and federated
		if(ColumnEncoderRecode.SORT_RECODE_MAP) {
			globalEncoder.applyToAll(ColumnEncoderRecode.class, ColumnEncoderRecode::sortCPRecodeMaps);
		}

		FrameBlock meta = new FrameBlock((int) fin.getNumColumns(), Types.ValueType.STRING);
		meta.setColumnNames(colNames);
		globalEncoder.getMetaData(meta);
		globalEncoder.initMetaData(meta);

		encodeFederatedFrames(fedMapping, globalEncoder, ec.getMatrixObject(getOutput(0)));

		// release input and outputs
		ec.setFrameOutput(getOutput(1).getName(), _metaReturn ? meta : new FrameBlock());
	}

	private class EncoderColnames {
		public final MultiColumnEncoder _encoder;
		public final String[] _colnames;

		public EncoderColnames(MultiColumnEncoder encoder, String[] colnames) {
			_encoder = encoder;
			_colnames = colnames;
		}
	}

	public EncoderColnames createGlobalEncoderWithEquiHeight(ExecutionContext ec, FrameObject fin, String spec) {
		// the encoder in which the complete encoding information will be aggregated
		MultiColumnEncoder globalEncoder = new MultiColumnEncoder(new ArrayList<>());
		String[] colNames = new String[(int) fin.getNumColumns()];

		Map<Integer, double[]> quantilesPerColumn = new HashMap<>();
		FederationMap fedMapping = fin.getFedMapping();
		fedMapping.forEachParallel((range, data) -> {
			int columnOffset = (int) range.getBeginDims()[1];

			// create an encoder with the given spec. The columnOffset (which is 0 based) has to be used to
			// tell the federated worker how much the indexes in the spec have to be offset.
			Future<FederatedResponse> responseFuture = data.executeFederatedOperation(
				new FederatedRequest(RequestType.EXEC_UDF, -1,
					new CreateFrameEncoder(data.getVarID(), spec, columnOffset + 1)));
			// collect responses with encoders
			try {
				FederatedResponse response = responseFuture.get();
				MultiColumnEncoder encoder = (MultiColumnEncoder) response.getData()[0];

				// put columns to equi-height
				for(Encoder enc : encoder.getColumnEncoders()) {
					if(enc instanceof ColumnEncoderComposite) {
						for(Encoder compositeEncoder : ((ColumnEncoderComposite) enc).getEncoders()) {
							if(compositeEncoder instanceof ColumnEncoderBin && ((ColumnEncoderBin) compositeEncoder).getBinMethod() == ColumnEncoderBin.BinMethod.EQUI_HEIGHT) {
								double quantilrRange = (double) fin.getNumRows() / ((ColumnEncoderBin) compositeEncoder).getNumBin();
								double[] quantiles = new double[((ColumnEncoderBin) compositeEncoder).getNumBin()];
								for(int i = 0; i < quantiles.length; i++) {
									quantiles[i] = quantilrRange * (i + 1);
								}
								quantilesPerColumn.put(((ColumnEncoderBin) compositeEncoder).getColID() + columnOffset - 1, quantiles);
							}
						}
					}
				}

				// merge this encoder into a composite encoder
				synchronized(globalEncoder) {
					globalEncoder.mergeAt(encoder, columnOffset, (int) (range.getBeginDims()[0] + 1));
				}
				// no synchronization necessary since names should anyway match
				String[] subRangeColNames = (String[]) response.getData()[1];
				System.arraycopy(subRangeColNames, 0, colNames, (int) range.getBeginDims()[1], subRangeColNames.length);
			}
			catch(Exception e) {
				throw new DMLRuntimeException("Federated encoder creation failed: ", e);
			}
			return null;
		});

		// calculate all quantiles
		Map<Integer, double[]> equiHeightBinsPerColumn = new HashMap<>();
		for(Map.Entry<Integer, double[]> colQuantiles : quantilesPerColumn.entrySet()) {
			QuantilePickFEDInstruction quantileInstr = new QuantilePickFEDInstruction(
				null, input1, output, PickByCount.OperationTypes.VALUEPICK,true, "qpick", "");
			MatrixBlock quantiles = quantileInstr.getEquiHeightBins(ec, colQuantiles.getKey(), colQuantiles.getValue());
			equiHeightBinsPerColumn.put(colQuantiles.getKey(), quantiles.getDenseBlockValues());
		}

		// modify global encoder
		for(Encoder enc : globalEncoder.getColumnEncoders()) {
			if(enc instanceof ColumnEncoderComposite) {
				for(Encoder compositeEncoder : ((ColumnEncoderComposite) enc).getEncoders())
					if(compositeEncoder instanceof ColumnEncoderBin && ((ColumnEncoderBin) compositeEncoder)
						.getBinMethod() == ColumnEncoderBin.BinMethod.EQUI_HEIGHT)
						((ColumnEncoderBin) compositeEncoder).build(null, equiHeightBinsPerColumn
							.get(((ColumnEncoderBin) compositeEncoder).getColID() - 1));
				((ColumnEncoderComposite) enc).updateAllDCEncoders();
			}
		}
		return new EncoderColnames(globalEncoder, colNames);
	}

	public static void encodeFederatedFrames(FederationMap fedMapping, MultiColumnEncoder globalencoder,
		MatrixObject transformedMat) {
		long varID = FederationUtils.getNextFedDataID();
		LongAdder nnz = new LongAdder();
		FederationMap tfFedMap = fedMapping.mapParallel(varID, (range, data) -> {
			// copy because we reuse it
			long[] beginDims = range.getBeginDims();
			long[] endDims = range.getEndDims();
			IndexRange ixRange = new IndexRange(beginDims[0], endDims[0], beginDims[1], endDims[1]).add(1);
			IndexRange ixRangeInv = new IndexRange(0, beginDims[0], 0, beginDims[1]);

			// get the encoder segment that is relevant for this federated worker
			MultiColumnEncoder encoder = globalencoder.subRangeEncoder(ixRange);
			// update begin end dims (column part) considering columns added by dummycoding
			encoder.updateIndexRanges(beginDims, endDims, globalencoder.getNumExtraCols(ixRangeInv));

			try {
				FederatedResponse response = data.executeFederatedOperation(
					new FederatedRequest(RequestType.EXEC_UDF,
					-1, new ExecuteFrameEncoder(data.getVarID(), varID, encoder))).get();
				if(!response.isSuccessful())
					response.throwExceptionFromResponse();
				nnz.add((Long)response.getData()[0]);
			}
			catch(Exception e) {
				throw new DMLRuntimeException(e);
			}
			return null;
		});

		// construct a federated matrix with the encoded data
		transformedMat.getDataCharacteristics()
			.setDimension(tfFedMap.getMaxIndexInRange(0), tfFedMap.getMaxIndexInRange(1))
			.setNonZeros(nnz.longValue());
		transformedMat.setFedMapping(tfFedMap);
	}

	public static class CreateFrameEncoder extends FederatedUDF {
		private static final long serialVersionUID = 2376756757742169692L;
		private final String _spec;
		private final int _offset;

		public CreateFrameEncoder(long input, String spec, int offset) {
			super(new long[] {input});
			_spec = spec;
			_offset = offset;
		}

		@Override
		public FederatedResponse execute(ExecutionContext ec, Data... data) {
			FrameObject fo = (FrameObject) data[0];
			FrameBlock fb = fo.acquireRead();
			String[] colNames = fb.getColumnNames();

			// create the encoder
			MultiColumnEncoder encoder = EncoderFactory
				.createEncoder(_spec, colNames, fb.getNumColumns(), null, _offset, _offset + fb.getNumColumns());

			// build necessary structures for encoding
			//encoder.build(fb, OptimizerUtils.getTransformNumThreads()); // FIXME skip equi-height sorting
			// FIXME: Enabling multithreading intermittently hangs
			encoder.build(fb, 1);
			fo.release();

			// create federated response
			return new FederatedResponse(ResponseType.SUCCESS, new Object[] {encoder, fb.getColumnNames()});
		}

		@Override
		public Pair<String, LineageItem> getLineageItem(ExecutionContext ec) {
			return null;
		}
	}

	public static class ExecuteFrameEncoder extends FederatedUDF {
		private static final long serialVersionUID = 6034440964680578276L;
		private final long _outputID;
		private final MultiColumnEncoder _encoder;

		public ExecuteFrameEncoder(long input, long output, MultiColumnEncoder encoder) {
			super(new long[] {input});
			_outputID = output;
			_encoder = encoder;
		}

		@Override
		public FederatedResponse execute(ExecutionContext ec, Data... data) {
			try{

				FrameBlock fb = ((FrameObject) data[0]).acquireReadAndRelease();
	
				// offset is applied on the Worker to shift the local encoders to their respective column
				_encoder.applyColumnOffset();
				// apply transformation
				MatrixBlock mbout = _encoder.apply(fb, OptimizerUtils.getTransformNumThreads());
				// FIXME: Enabling multithreading intermittently hangs
				// MatrixBlock mbout = _encoder.apply(fb, 1);
	
				// create output matrix object
				MatrixObject mo = ExecutionContext.createMatrixObject(mbout);
	
				// add it to the list of variables
				ec.setVariable(String.valueOf(_outputID), mo);
	
				// return id handle
				return new FederatedResponse(
					ResponseType.SUCCESS_EMPTY, mbout.getNonZeros());
			}
			catch(Exception e){
				return new FederatedResponse(ResponseType.ERROR);
			}
		}

		@Override
		public List<Long> getOutputIds() {
			return new ArrayList<>(Arrays.asList(_outputID));
		}

		@Override
		public Pair<String, LineageItem> getLineageItem(ExecutionContext ec) {
			LineageItem[] liUdfInputs = Arrays.stream(getInputIDs())
				.mapToObj(id -> ec.getLineage().get(String.valueOf(id))).toArray(LineageItem[]::new);
			// calculate checksum for the encoder
			Checksum checksum = new Adler32();
			byte[] bytes = SerializationUtils.serialize(_encoder);
			checksum.update(bytes, 0, bytes.length);
			CPOperand encoder = new CPOperand(String.valueOf(checksum.getValue()), ValueType.INT64, DataType.SCALAR,
				true);
			LineageItem[] otherInputs = LineageItemUtils.getLineage(ec, encoder);
			LineageItem[] liInputs = Stream.concat(Arrays.stream(liUdfInputs), Arrays.stream(otherInputs))
				.toArray(LineageItem[]::new);
			return Pair.of(String.valueOf(_outputID), new LineageItem(getClass().getSimpleName(), liInputs));
		}
	}
}
