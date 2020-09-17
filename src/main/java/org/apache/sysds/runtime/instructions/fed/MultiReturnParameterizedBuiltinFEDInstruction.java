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
import java.util.concurrent.Future;

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.FrameObject;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest.RequestType;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse.ResponseType;
import org.apache.sysds.runtime.controlprogram.federated.FederatedUDF;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse;
import org.apache.sysds.runtime.controlprogram.federated.FederationMap;
import org.apache.sysds.runtime.controlprogram.federated.FederationUtils;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.transform.encode.Encoder;
import org.apache.sysds.runtime.transform.encode.EncoderBin;
import org.apache.sysds.runtime.transform.encode.EncoderComposite;
import org.apache.sysds.runtime.transform.encode.EncoderDummycode;
import org.apache.sysds.runtime.transform.encode.EncoderFactory;
import org.apache.sysds.runtime.transform.encode.EncoderFeatureHash;
import org.apache.sysds.runtime.transform.encode.EncoderMVImpute;
import org.apache.sysds.runtime.transform.encode.EncoderOmit;
import org.apache.sysds.runtime.transform.encode.EncoderPassThrough;
import org.apache.sysds.runtime.transform.encode.EncoderRecode;
import org.apache.sysds.runtime.util.IndexRange;

public class MultiReturnParameterizedBuiltinFEDInstruction extends ComputationFEDInstruction {
	protected final ArrayList<CPOperand> _outputs;

	private MultiReturnParameterizedBuiltinFEDInstruction(Operator op, CPOperand input1, CPOperand input2,
		ArrayList<CPOperand> outputs, String opcode, String istr) {
		super(FEDType.MultiReturnParameterizedBuiltin, op, input1, input2, null, opcode, istr);
		_outputs = outputs;
	}

	public CPOperand getOutput(int i) {
		return _outputs.get(i);
	}

	public static MultiReturnParameterizedBuiltinFEDInstruction parseInstruction(String str) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		ArrayList<CPOperand> outputs = new ArrayList<>();
		String opcode = parts[0];

		if(opcode.equalsIgnoreCase("transformencode")) {
			// one input and two outputs
			CPOperand in1 = new CPOperand(parts[1]);
			CPOperand in2 = new CPOperand(parts[2]);
			outputs.add(new CPOperand(parts[3], Types.ValueType.FP64, Types.DataType.MATRIX));
			outputs.add(new CPOperand(parts[4], Types.ValueType.STRING, Types.DataType.FRAME));
			return new MultiReturnParameterizedBuiltinFEDInstruction(null, in1, in2, outputs, opcode, str);
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
		EncoderComposite globalEncoder = new EncoderComposite(
			// IMPORTANT: Encoder order matters
			Arrays.asList(new EncoderRecode(),
				new EncoderFeatureHash(),
				new EncoderPassThrough(),
				new EncoderBin(),
				new EncoderDummycode(),
				new EncoderOmit(true),
				new EncoderMVImpute()));
		// first create encoders at the federated workers, then collect them and aggregate them to a single large
		// encoder
		FederationMap fedMapping = fin.getFedMapping();
		fedMapping.forEachParallel((range, data) -> {
			int columnOffset = (int) range.getBeginDims()[1] + 1;

			// create an encoder with the given spec. The columnOffset (which is 1 based) has to be used to
			// tell the federated worker how much the indexes in the spec have to be offset.
			Future<FederatedResponse> responseFuture = data.executeFederatedOperation(
				new FederatedRequest(RequestType.EXEC_UDF, -1,
					new CreateFrameEncoder(data.getVarID(), spec, columnOffset)));
			// collect responses with encoders
			try {
				FederatedResponse response = responseFuture.get();
				Encoder encoder = (Encoder) response.getData()[0];
				// merge this encoder into a composite encoder
				synchronized(globalEncoder) {
					globalEncoder.mergeAt(encoder, (int) (range.getBeginDims()[0] + 1), columnOffset);
				}
				// no synchronization necessary since names should anyway match
				String[] subRangeColNames = (String[]) response.getData()[1];
				System.arraycopy(subRangeColNames, 0, colNames, (int) range.getBeginDims()[1], subRangeColNames.length);
			}
			catch(Exception e) {
				throw new DMLRuntimeException("Federated encoder creation failed: " + e.getMessage());
			}
			return null;
		});
		FrameBlock meta = new FrameBlock((int) fin.getNumColumns(), Types.ValueType.STRING);
		meta.setColumnNames(colNames);
		globalEncoder.getMetaData(meta);
		globalEncoder.initMetaData(meta);

		encodeFederatedFrames(fedMapping, globalEncoder, ec.getMatrixObject(getOutput(0)));
		
		// release input and outputs
		ec.setFrameOutput(getOutput(1).getName(), meta);
	}
	
	public static void encodeFederatedFrames(FederationMap fedMapping, Encoder globalEncoder,
		MatrixObject transformedMat) {
		long varID = FederationUtils.getNextFedDataID();
		FederationMap transformedFedMapping = fedMapping.mapParallel(varID, (range, data) -> {
			// copy because we reuse it
			long[] beginDims = range.getBeginDims();
			long[] endDims = range.getEndDims();
			IndexRange ixRange = new IndexRange(beginDims[0], endDims[0], beginDims[1], endDims[1]).add(1);// make 1-based

			// update begin end dims (column part) considering columns added by dummycoding
			globalEncoder.updateIndexRanges(beginDims, endDims);

			// get the encoder segment that is relevant for this federated worker
			Encoder encoder = globalEncoder.subRangeEncoder(ixRange);

			try {
				FederatedResponse response = data.executeFederatedOperation(new FederatedRequest(RequestType.EXEC_UDF,
					-1, new ExecuteFrameEncoder(data.getVarID(), varID, encoder))).get();
				if(!response.isSuccessful())
					response.throwExceptionFromResponse();
			}
			catch(Exception e) {
				throw new DMLRuntimeException(e);
			}
			return null;
		});

		// construct a federated matrix with the encoded data
		transformedMat.getDataCharacteristics().setDimension(
			transformedFedMapping.getMaxIndexInRange(0), transformedFedMapping.getMaxIndexInRange(1));
		transformedMat.setFedMapping(transformedFedMapping);
	}
	
	public static class CreateFrameEncoder extends FederatedUDF {
		private static final long serialVersionUID = 2376756757742169692L;
		private final String _spec;
		private final int _offset;
		
		public CreateFrameEncoder(long input, String spec, int offset) {
			super(new long[]{input});
			_spec = spec;
			_offset = offset;
		}

		@Override
		public FederatedResponse execute(ExecutionContext ec, Data... data) {
			FrameObject fo = (FrameObject) data[0];
			FrameBlock fb = fo.acquireRead();
			String[] colNames = fb.getColumnNames();

			// create the encoder
			Encoder encoder = EncoderFactory.createEncoder(_spec, colNames,
				fb.getNumColumns(), null, _offset, _offset + fb.getNumColumns());
			
			// build necessary structures for encoding
			encoder.build(fb);
			fo.release();

			// create federated response
			return new FederatedResponse(ResponseType.SUCCESS, new Object[] {encoder, fb.getColumnNames()});
		}
	}

	public static class ExecuteFrameEncoder extends FederatedUDF {
		private static final long serialVersionUID = 6034440964680578276L;
		private final long _outputID;
		private final Encoder _encoder;
		
		public ExecuteFrameEncoder(long input, long output, Encoder encoder) {
			super(new long[] {input});
			_outputID = output;
			_encoder = encoder;
		}

		@Override
		public FederatedResponse execute(ExecutionContext ec, Data... data) {
			FrameBlock fb = ((FrameObject)data[0]).acquireReadAndRelease();

			// apply transformation
			MatrixBlock mbout = _encoder.apply(fb,
				new MatrixBlock(fb.getNumRows(), fb.getNumColumns(), false));

			// create output matrix object
			MatrixObject mo = ExecutionContext.createMatrixObject(mbout);

			// add it to the list of variables
			ec.setVariable(String.valueOf(_outputID), mo);
		
			// return id handle
			return new FederatedResponse(ResponseType.SUCCESS_EMPTY);
		}
	}
}
