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

import java.util.Arrays;
import java.util.Collections;
import java.util.concurrent.Future;
import java.util.Iterator;
import java.util.SortedMap;
import java.util.stream.IntStream;
import java.util.TreeMap;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRange;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse;
import org.apache.sysds.runtime.controlprogram.federated.FederatedUDF;
import org.apache.sysds.runtime.controlprogram.federated.FederationMap;
import org.apache.sysds.runtime.controlprogram.federated.FederationMap.AlignType;
import org.apache.sysds.runtime.controlprogram.federated.FederationUtils;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;

public class CtableFEDInstruction extends ComputationFEDInstruction {
	private final CPOperand _outDim1;
	private final CPOperand _outDim2;
	//private final boolean _isExpand;
	//private final boolean _ignoreZeros;

	private CtableFEDInstruction(CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out, String outputDim1, boolean dim1Literal, String outputDim2, boolean dim2Literal, boolean isExpand,
		boolean ignoreZeros, String opcode, String istr) {
		super(FEDType.Ctable, null, in1, in2, in3, out, opcode, istr);
		_outDim1 = new CPOperand(outputDim1, ValueType.FP64, DataType.SCALAR, dim1Literal);
		_outDim2 = new CPOperand(outputDim2, ValueType.FP64, DataType.SCALAR, dim2Literal);
		//_isExpand = isExpand;
		//_ignoreZeros = ignoreZeros;
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

		boolean reversed = false;
		if(!mo1.isFederated() && mo2.isFederated()) {
			mo1 = ec.getMatrixObject(input2);
			mo2 = ec.getMatrixObject(input1);
			reversed = true;
		}

		// get new output dims
		Long[] dims1 = getOutputDimension(mo1, reversed ? input2 : input1, reversed ? _outDim2 : _outDim1,
			mo1.getFedMapping().getFederatedRanges());
		Long[] dims2 = getOutputDimension(mo2, reversed ? input1 : input2, reversed ? _outDim1 : _outDim2,
			mo1.getFedMapping().getFederatedRanges());

		MatrixObject mo3 = input3 != null && input3.isMatrix() ? ec.getMatrixObject(input3) : null;

		boolean reversedWeights = mo3 != null && mo3.isFederated() && !(mo1.isFederated() || mo2.isFederated());
		if(reversedWeights) {
			mo3 = mo1;
			mo1 = ec.getMatrixObject(input3);
		}

		boolean fedOutput = isFedOutput(mo1.getFedMapping(), mo2);

		processRequest(ec, mo1, mo2, mo3, reversed, reversedWeights, fedOutput, dims1, dims2);
	}

	private void processRequest(ExecutionContext ec, MatrixObject mo1, MatrixObject mo2, MatrixObject mo3,
		boolean reversed, boolean reversedWeights, boolean fedOutput, Long[] dims1, Long[] dims2) {

		FederationMap fedMap = mo1.getFedMapping();
		// static non-partitioned output dimension (same for all federated partitions)
		long staticDim = Collections.max(Arrays.asList(dims1), Long::compare);

		FederatedRequest[] fr1 = fedMap.broadcastSliced(mo2, false);
		FederatedRequest[] fr2 = null;
		FederatedRequest fr3, fr4, fr5, fr6, fr7;
		fr3 = fr4 = fr5 = fr6 = fr7 = null;
		Future<FederatedResponse>[] ffr = null;

		if(mo3 != null && mo1.isFederated() && mo3.isFederated()
			&& fedMap.isAligned(mo3.getFedMapping(), AlignType.FULL)) { // mo1 and mo3 federated and aligned
			if(!reversed)
				fr3 = FederationUtils.callInstruction(instString, output, new CPOperand[] {input1, input2, input3},
					new long[] {fedMap.getID(), fr1[0].getID(), mo3.getFedMapping().getID()});
			else
				fr3 = FederationUtils.callInstruction(instString, output, new CPOperand[] {input1, input2, input3},
					new long[] {fr1[0].getID(), fedMap.getID(), mo3.getFedMapping().getID()});

			fr5 = fedMap.cleanup(getTID(), fr1[0].getID());
		}
		else if(mo3 == null) {
			if(!reversed)
				fr3 = FederationUtils.callInstruction(instString, output, new CPOperand[] {input1, input2},
					new long[] {fedMap.getID(), fr1[0].getID()});
			else
				fr3 = FederationUtils.callInstruction(instString, output, new CPOperand[] {input1, input2},
					new long[] {fr1[0].getID(), fedMap.getID()});

			fr5 = fedMap.cleanup(getTID(), fr1[0].getID());
		}
		else {
			fr2 = fedMap.broadcastSliced(mo3, false);
			if(!reversed && !reversedWeights)
				fr3 = FederationUtils.callInstruction(instString, output, new CPOperand[] {input1, input2, input3},
					new long[] {fedMap.getID(), fr1[0].getID(), fr2[0].getID()});
			else if(reversed && !reversedWeights)
				fr3 = FederationUtils.callInstruction(instString, output, new CPOperand[] {input1, input2, input3},
					new long[] {fr1[0].getID(), fedMap.getID(), fr2[0].getID()});
			else
				fr3 = FederationUtils.callInstruction(instString, output, new CPOperand[] {input1, input2, input3},
					new long[] {fr1[0].getID(), fr2[0].getID(), fedMap.getID()});

			fr5 = fedMap.cleanup(getTID(), fr1[0].getID());
			fr6 = fedMap.cleanup(getTID(), fr2[0].getID());
		}

		if(fedOutput) {
			if(fr2 != null && fr6 != null) // broadcasted mo3
				fedMap.execute(getTID(), true, fr1, fr2, fr3, fr5, fr6);
			else
				fedMap.execute(getTID(), true, fr1, fr3, fr5);

			MatrixObject out = ec.getMatrixObject(output);
			FederationMap newFedMap = modifyFedRanges(fedMap.copyWithNewID(fr3.getID()),
				staticDim, dims2, reversed);
			setFedOutput(mo1, out, newFedMap, staticDim, dims2, fr3.getID(), reversed);
		} else {
			fr4 = new FederatedRequest(FederatedRequest.RequestType.GET_VAR, fr3.getID());
			fr7 = fedMap.cleanup(getTID(), fr3.getID());
			if(fr2 != null && fr6 != null) // broadcasted mo3
				ffr = fedMap.execute(getTID(), true, fr1, fr2, fr3, fr4, fr5, fr6, fr7);
			else
				ffr = fedMap.execute(getTID(), true, fr1, fr3, fr4, fr5, fr7);

			ec.setMatrixOutput(output.getName(), aggResult(ffr));
		}
	}

	private boolean isFedOutput(FederationMap fedMap, MatrixObject mo2) {
		MatrixBlock mb = mo2.acquireReadAndRelease();
		FederatedRange[] fedRanges = fedMap.getFederatedRanges(); // federated ranges of mo1
		SortedMap<Double, Double> fedDims = new TreeMap<Double, Double>(); // <beginDim, endDim>

		// collect min and max of the corresponding slices of mo2
		IntStream.range(0, fedRanges.length).forEach(i -> {
			MatrixBlock sliced = mb.slice(
				fedRanges[i].getBeginDimsInt()[0], fedRanges[i].getEndDimsInt()[0] - 1,
				fedRanges[i].getBeginDimsInt()[1], fedRanges[i].getEndDimsInt()[1] - 1);
			fedDims.put(sliced.min(), sliced.max());
		});

		boolean retVal = (fedDims.size() == fedRanges.length); // no duplicate begin dimension entries

		Iterator<SortedMap.Entry<Double, Double>> iter = fedDims.entrySet().iterator();
		SortedMap.Entry<Double, Double> entry = iter.next(); // first entry does not have to be checked
		double prevEndDim = entry.getValue().doubleValue();
		while(iter.hasNext() && retVal) {
			entry = iter.next();
			// previous end dimension must be less than current begin dimension (no overlaps of ranges)
			retVal &= (prevEndDim < entry.getKey());
			prevEndDim = entry.getValue().doubleValue();
		}

		return retVal;
	}

	private static void setFedOutput(MatrixObject mo1, MatrixObject out, FederationMap fedMap,
		long staticDim, Long[] dims2, long outId, boolean reversed) {
		// get the final output dimensions
		final long d1 = (reversed ? Collections.max(Arrays.asList(dims2)) : staticDim);
		final long d2 = (reversed ? staticDim : Collections.max(Arrays.asList(dims2)));

		// set output
		out.getDataCharacteristics().set(d1, d2, (int) mo1.getBlocksize(), mo1.getNnz());
		out.setFedMapping(fedMap);

		long varID = FederationUtils.getNextFedDataID();
		fedMap.mapParallel(varID, (range, data) -> {
			try {
				FederatedResponse response = data.executeFederatedOperation(new FederatedRequest(
					FederatedRequest.RequestType.EXEC_UDF, -1,
					new SliceOutput(data.getVarID(), staticDim, dims2, reversed))).get();
				if(!response.isSuccessful())
					response.throwExceptionFromResponse();
			}
			catch(Exception e) {
				throw new DMLRuntimeException(e);
			}
			return null;
		});
	}

	private static MatrixBlock aggResult(Future<FederatedResponse>[] ffr) {
		MatrixBlock resultBlock = new MatrixBlock(1, 1, true, 0);
		int dim1 = 0, dim2 = 0;
		for(int i = 0; i < ffr.length; i++) {
			try {
				MatrixBlock mb = ((MatrixBlock) ffr[i].get().getData()[0]);
				dim1 = mb.getNumRows()  > dim1 ? mb.getNumRows() : dim1;
				dim2 = mb.getNumColumns()  > dim2 ? mb.getNumColumns() : dim2;

				// set next and prev to same output dimensions
				MatrixBlock prev = new MatrixBlock(dim1, dim2, true, 0);
				prev.copy(0, resultBlock.getNumRows()-1, 0, resultBlock.getNumColumns()-1, resultBlock, true);

				MatrixBlock next = new MatrixBlock(dim1, dim2, true, 0);
				next.copy(0, mb.getNumRows()-1, 0, mb.getNumColumns()-1, mb, true);

				// add worker results
				BinaryOperator plus = InstructionUtils.parseBinaryOperator("+");
				resultBlock = prev.binaryOperationsInPlace(plus, next);
			}
			catch(Exception e) {
				e.printStackTrace();
			}
		}
		return resultBlock;
	}

	private static FederationMap modifyFedRanges(FederationMap fedMap, long staticDim,
		Long[] dims2, boolean reversed) {
		// set the federated ranges to the individual partition sizes
		IntStream.range(0, fedMap.getFederatedRanges().length).forEach(counter -> {
			FederatedRange fedRange = fedMap.getFederatedRanges()[counter];
			fedRange.setBeginDim(reversed ? 1 : 0, 0);
			fedRange.setEndDim(reversed ? 1 : 0, staticDim);
			fedRange.setBeginDim(reversed ? 0 : 1, counter == 0 ? 0 : dims2[counter-1]);
			fedRange.setEndDim(reversed ? 0 : 1, dims2[counter]);
		});
		return fedMap;
	}

	private Long[] getOutputDimension(MatrixObject in, CPOperand inOp, CPOperand outOp, FederatedRange[] federatedRanges) {
		Long[] fedDims = new Long[federatedRanges.length];

		if(!in.isFederated()) {
			//slice
			MatrixBlock mb = in.acquireReadAndRelease();
			IntStream.range(0, federatedRanges.length).forEach(i -> {
				MatrixBlock sliced = mb.slice(
					federatedRanges[i].getBeginDimsInt()[0], federatedRanges[i].getEndDimsInt()[0] - 1,
					federatedRanges[i].getBeginDimsInt()[1], federatedRanges[i].getEndDimsInt()[1] - 1);
				fedDims[i] = (long) sliced.max();
			});
			return fedDims;
		}

		String maxInstString = constructMaxInstString(inOp.getName(), outOp.getName());

		// get max per worker
		FederationMap map = in.getFedMapping();
		FederatedRequest fr1 = FederationUtils.callInstruction(maxInstString, outOp,
			new CPOperand[]{inOp}, new long[]{in.getFedMapping().getID()});
		FederatedRequest fr2 = new FederatedRequest(FederatedRequest.RequestType.GET_VAR, fr1.getID());
		FederatedRequest fr3 = map.cleanup(getTID(), fr1.getID());
		Future<FederatedResponse>[] tmp = map.execute(getTID(), fr1, fr2, fr3);

		return computeOutputDims(tmp);
	}

	private static Long[] computeOutputDims(Future<FederatedResponse>[] tmp) {
		Long[] fedDims = new Long[tmp.length];
		for(int i = 0; i < tmp.length; i ++)
			try {
				fedDims[i] = ((ScalarObject) tmp[i].get().getData()[0]).getLongValue();
			}
			catch(Exception e) {
				e.printStackTrace();
			}
		return fedDims;
	}

	private String constructMaxInstString(String in, String out) {
		String maxInstrString = instString.replace("ctable", "uamax");
		String[] instParts = maxInstrString.split(Lop.OPERAND_DELIMITOR);
		String[] maxInstParts = new String[] {instParts[0], instParts[1],
			InstructionUtils.concatOperandParts(in, DataType.MATRIX.name(), (ValueType.FP64).name()),
			InstructionUtils.concatOperandParts(out, DataType.SCALAR.name(), (ValueType.FP64).name()), "16"};
		return String.join(Lop.OPERAND_DELIMITOR, maxInstParts);
	}

	private static class SliceOutput extends FederatedUDF {

		private static final long serialVersionUID = -2808597461054603816L;
		private final int _staticDim;
		private final Long[] _fedDims;
		private final boolean _reversed;

		protected SliceOutput(long input, long staticDim, Long[] fedDims, boolean reversed) {
			super(new long[] {input});
			_staticDim = (int)staticDim;
			_fedDims = fedDims;
			_reversed = reversed;
		}

		public FederatedResponse execute(ExecutionContext ec, Data... data) {
			MatrixObject mo = (MatrixObject) data[0];
			MatrixBlock mb = mo.acquireReadAndRelease();

			int beginDim = 0;
			int endDim = (_reversed ? mb.getNumRows() : mb.getNumColumns());
			int localStaticDim = (_reversed ? mb.getNumColumns() : mb.getNumRows());
			for(int counter = 0; counter < _fedDims.length; counter++) {
				if(_fedDims[counter] == endDim) {
					beginDim = (counter == 0 ? 0 : _fedDims[counter - 1].intValue());
					break;
				}
			}

			mb = expandMatrix(mb, localStaticDim);

			// crop the output
			MatrixBlock sliced = _reversed ? mb.slice(beginDim, endDim - 1, 0, _staticDim - 1)
				: mb.slice(0, _staticDim - 1, beginDim, endDim - 1);
			mo.acquireModify(sliced);
			mo.release();

			return new FederatedResponse(FederatedResponse.ResponseType.SUCCESS, new Object[] {});
		}

		private MatrixBlock expandMatrix(MatrixBlock mb, int localStaticDim) {
			int diff = _staticDim - localStaticDim;
			if(diff > 0) {
				MatrixBlock tmpMb = (_reversed ? new MatrixBlock(mb.getNumRows(), diff, (double) 0)
					: new MatrixBlock(diff, mb.getNumColumns(), (double) 0));
				mb = mb.append(tmpMb, null, _reversed);
			}
			return mb;
		}

		@Override
		public Pair<String, LineageItem> getLineageItem(ExecutionContext ec) {
			return null;
		}
	}
}
