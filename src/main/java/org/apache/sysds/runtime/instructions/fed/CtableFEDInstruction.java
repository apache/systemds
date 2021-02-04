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
import java.util.Collections;
import java.util.List;
import java.util.concurrent.Future;
import java.util.stream.IntStream;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.sysds.api.mlcontext.Matrix;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.lops.Ctable;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRange;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse;
import org.apache.sysds.runtime.controlprogram.federated.FederatedUDF;
import org.apache.sysds.runtime.controlprogram.federated.FederationMap;
import org.apache.sysds.runtime.controlprogram.federated.FederationUtils;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.ReduceAll;
import org.apache.sysds.runtime.functionobjects.ValueFunction;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.instructions.cp.DoubleObject;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.AggregateOperator;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;

public class CtableFEDInstruction extends ComputationFEDInstruction {
	private final CPOperand _outDim1;
	private final CPOperand _outDim2;
	private final boolean _isExpand;
	private final boolean _ignoreZeros;

	private CtableFEDInstruction(CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out, String outputDim1, boolean dim1Literal, String outputDim2, boolean dim2Literal, boolean isExpand,
		boolean ignoreZeros, String opcode, String istr) {
		super(FEDType.Ctable, null, in1, in2, in3, out, opcode, istr);
		_outDim1 = new CPOperand(outputDim1, ValueType.FP64, DataType.SCALAR, dim1Literal);
		_outDim2 = new CPOperand(outputDim2, ValueType.FP64, DataType.SCALAR, dim2Literal);
		_isExpand = isExpand;
		_ignoreZeros = ignoreZeros;
	}

	public static CtableFEDInstruction parseInstruction(String inst) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(inst);
		InstructionUtils.checkNumFields(parts, 7);

		String opcode = parts[0];

		//handle opcode
		if(!(opcode.equalsIgnoreCase("ctable"))) {
			throw new DMLRuntimeException("Unexpected opcode in CtableFEDInstruction: " + inst);
		}

		//handle operands
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand in2 = new CPOperand(parts[2]);
		CPOperand in3 = new CPOperand(parts[3]);

		//handle known dimension information
		String[] dim1Fields = parts[4].split(Instruction.LITERAL_PREFIX);
		String[] dim2Fields = parts[5].split(Instruction.LITERAL_PREFIX);

		CPOperand out = new CPOperand(parts[6]);
		boolean ignoreZeros = Boolean.parseBoolean(parts[7]);

		// ctable does not require any operator, so we simply pass-in a dummy operator with null functionobject
		return new CtableFEDInstruction(in1,
			in2, in3, out, dim1Fields[0], Boolean.parseBoolean(dim1Fields[1]),
			dim2Fields[0], Boolean.parseBoolean(dim2Fields[1]), false, ignoreZeros, opcode, inst);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		MatrixObject mo1 = ec.getMatrixObject(input1);
		MatrixObject mo2 = ec.getMatrixObject(input2);

		//TODO if empty etc, also scalar dims?
		MatrixObject mo3 = input3.isScalar() ? null : ec.getMatrixObject(input3);

//		MatrixObject out = ec.getMatrixObject(output);

		//TODO if output dim is defined
		long outputDim1 = ec.getScalarInput(_outDim1).getLongValue();
		long outputDim2 = ec.getScalarInput(_outDim2).getLongValue();

		boolean outputDimsKnown = (outputDim1 != -1 && outputDim2 != -1);

		//canonicalization for federated lhs
		//		if( !mo1.isFederated() && mo2.isFederated()
		//			&& mo1.getDataCharacteristics().equalDims(mo2.getDataCharacteristics())
		//			&& ((BinaryOperator)_optr).isCommutative() ) {
		//			mo1 = ec.getMatrixObject(input2);
		//			mo2 = ec.getMatrixObject(input1);
		//		}

		//construct commands: broadcast , fed ctable, clean broadcast
		FederatedRequest[] fr1 = mo1.getFedMapping().broadcastSliced(mo2, false);
		FederatedRequest fr2 = FederationUtils.callInstruction(instString, output, new CPOperand[] {input1, input2},
			new long[] {mo1.getFedMapping().getID(), fr1[0].getID()});
		FederatedRequest fr3 = mo1.getFedMapping().cleanup(getTID(), fr1[0].getID());
		mo1.getFedMapping().execute(getTID(), true, fr1, fr2, fr3);

		//TODO always same fed ranges
		long[] dims1 = getDimension(ec, input1, _outDim1, mo1.getFedMapping().getFederatedRanges());
		long[] dims2 = getDimension(ec, input2, _outDim2, mo1.getFedMapping().getFederatedRanges());

		outputDim1 = outputDimsKnown ? outputDim1 : Arrays.stream(dims1).max().getAsLong();
		outputDim2 = outputDimsKnown ? outputDim2 : Arrays.stream(dims2).max().getAsLong();

		MatrixObject out = ec.getMatrixObject(output);
		out.getDataCharacteristics().set(outputDim1, outputDim2, (int) mo1.getBlocksize());
		out.setFedMapping(mo1.getFedMapping().copyWithNewID(fr1[0].getID()));

//		MatrixBlock ret = out.acquireReadAndRelease();


		List<MatrixBlock> resMatrices = new ArrayList();
		FederationMap fedMap = out.getFedMapping();
		long varID = FederationUtils.getNextFedDataID();
		long finalOutputDim = outputDim1;
		long finalOutputDim1 = outputDim2;

		fedMap.mapParallel(varID, (range, data) -> {
			FederatedResponse response;
			try {
				response = data.executeFederatedOperation(new FederatedRequest(FederatedRequest.RequestType.EXEC_UDF,
					-1, new CtableFEDInstruction.MergeOutput(data.getVarID(), varID,
					new int[] {(int) finalOutputDim, (int) finalOutputDim1}))).get();
				if(!response.isSuccessful())
					response.throwExceptionFromResponse();

				resMatrices.add((MatrixBlock) response.getData()[0]);
			}
			catch(Exception e) {
				throw new DMLRuntimeException(e);
			}
			return null;
		});

		// merge results
		MatrixBlock resultBlock = new MatrixBlock((int) outputDim1, (int) outputDim2, false);
		for(MatrixBlock mb : resMatrices) {
			BinaryOperator plus = InstructionUtils.parseBinaryOperator("+");
			resultBlock = resultBlock.binaryOperationsInPlace(plus, mb);
		}

		ec.setMatrixOutput(output.getName(), resultBlock);
	}

	private long[] getDimension(ExecutionContext ec, CPOperand inOp, CPOperand outOp, FederatedRange[] federatedRanges) {
		MatrixObject in = ec.getMatrixObject(inOp);
		long[] fedDims = new long[federatedRanges.length];
		if(!in.isFederated()) {
			//slice
			MatrixBlock mb = ec.getMatrixInput(inOp.getName());
			IntStream.range(0, federatedRanges.length).forEach(i -> {
				MatrixBlock sliced = mb
					.slice(federatedRanges[i].getBeginDimsInt()[0], federatedRanges[i].getEndDimsInt()[0] - 1);
				fedDims[i] = (long) sliced.max();
			});
			return fedDims;
		}

		// construct new instString
		String maxInstrString = instString.replace("ctable", "uamax");
		String[] instParts = maxInstrString.split(Lop.OPERAND_DELIMITOR);
		String[] maxInstParts = new String[] {instParts[0], instParts[1],
			InstructionUtils.concatOperandParts(inOp.getName(), DataType.MATRIX.name(), (ValueType.FP64).name()),
			InstructionUtils.concatOperandParts(outOp.getName(), DataType.SCALAR.name(), (ValueType.FP64).name()), "16"};
		maxInstrString = String.join(Lop.OPERAND_DELIMITOR, maxInstParts);

		FederationMap map = in.getFedMapping();
		FederatedRequest fr1 = FederationUtils.callInstruction(maxInstrString, outOp,
			new CPOperand[]{inOp}, new long[]{in.getFedMapping().getID()});
		FederatedRequest fr2 = new FederatedRequest(FederatedRequest.RequestType.GET_VAR, fr1.getID());
		FederatedRequest fr3 = map.cleanup(getTID(), fr1.getID());
		Future<FederatedResponse>[] tmp = map.execute(getTID(), fr1, fr2, fr3);

		for(int i = 0; i < tmp.length; i ++) {
			try {
				fedDims[i] = ((ScalarObject) tmp[i].get().getData()[0]).getLongValue();
			}
			catch(Exception e) {
				e.printStackTrace();
			}
		}
		return fedDims;
	}

	public static class MergeOutput extends FederatedUDF {
		private static final long serialVersionUID = 1319031225775373602L;
		private final long _outputID;
		private final int[] _dims;

		public MergeOutput(long input, long outputID, int[] dims) {
			super(new long[] {input});
			_outputID = outputID;
			_dims = dims;
		}

		public FederatedResponse execute(ExecutionContext ec, Data... data) {
			MatrixObject mo = (MatrixObject) data[0];
			MatrixBlock mb = mo.acquireReadAndRelease();

			MatrixBlock res = res = new MatrixBlock(_dims[0], _dims[1], false);
			res.copy(0, mb.getNumRows()-1, 0, mb.getNumColumns()-1, mb, false);
			return new FederatedResponse(FederatedResponse.ResponseType.SUCCESS, new Object[] {res});
		}

		public Pair<String, LineageItem> getLineageItem(ExecutionContext ec) { return null; }
	}
}