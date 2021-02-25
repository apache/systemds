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
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.sysds.api.mlcontext.Matrix;
import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.lops.Ctable;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.parser.Statement;
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
import org.apache.sysds.runtime.instructions.cp.BooleanObject;
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

	//TODO check on parser level for slicefinder
	private boolean checkFedOutput(ExecutionContext ec) {
		String prog = ec.getProgram().getProgramBlocks().get(0).getStatementBlock().getStatements().stream().map(
			Statement::getText).collect(Collectors.joining("\n"));
		return prog.contains("fdom = colMaxs(X);") && prog.contains("foffb = t(cumsum(t(fdom))) - fdom;")
			&& prog.contains("foffe = t(cumsum(t(fdom)))\n")
			&& prog.contains("rix = matrix(seq(1,m)%*%matrix(1,1,n), m*n, 1)") && prog.contains("cix = matrix(X + foffb, m*n, 1);")
			&& prog.contains("table(rix, cix);");
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

		if(mo2.getNumColumns() != 1 && mo1.getNumColumns() != 1)
			throw new DMLRuntimeException("Federated ctable: Input vectors should be nx1.");

		if (checkFedOutput(ec))
			processFedOutputInstruction(ec, mo1, mo2, reversed);
		else
			processFedInstruction(ec, mo1, mo2, reversed);
	}

	private void processFedOutputInstruction(ExecutionContext ec, MatrixObject mo1, MatrixObject mo2, boolean reversed) {
		// modify instruction string by adding param to remove empty rows in fed parts
		// that are empty rows from other workers
		String newInstString = String.join(Lop.OPERAND_DELIMITOR, new String[] {instString, "true"});

		FederationMap fedMap = mo1.getFedMapping();
		FederatedRequest[] fr1 = fedMap.broadcastSliced(mo2, false);
		FederatedRequest fr2;
		if(reversed)
			fr2 = FederationUtils.callInstruction(newInstString, output, new CPOperand[]{input1, input2},
				new long[]{fr1[0].getID(), fedMap.getID()});
		else fr2 = FederationUtils.callInstruction(newInstString, output, new CPOperand[]{input1, input2},
			new long[]{fedMap.getID(), fr1[0].getID()});
		FederatedRequest fr3 = fedMap.cleanup(getTID(), fr1[0].getID());
		fedMap.execute(getTID(), true, fr1, fr2, fr3);

		// get new output dims
		long[] dims1 = getOutputDimension(ec, input1, _outDim1, mo1.getFedMapping().getFederatedRanges());
		long[] dims2 = getOutputDimension(ec, input2, _outDim2, mo1.getFedMapping().getFederatedRanges());

		// modify fed ranges
		for(int i = 0; i < fedMap.getFederatedRanges().length; i++) {
			fedMap.getFederatedRanges()[i].setBeginDim(0, i == 0 ? 0 : fedMap.getFederatedRanges()[i-1].getEndDims()[0]);
			fedMap.getFederatedRanges()[i].setEndDim(0, dims1[i]);

			fedMap.getFederatedRanges()[i].setBeginDim(1, i == 0 ? 0 : fedMap.getFederatedRanges()[i-1].getBeginDims()[1]);
			fedMap.getFederatedRanges()[i].setEndDim(1, dims2[i]);
		}

		// set output
		MatrixObject out = ec.getMatrixObject(output);
		out.getDataCharacteristics().set(Arrays.stream(dims1).max().getAsLong(), Arrays.stream(dims2).max().getAsLong(),
			(int) mo1.getBlocksize(), mo1.getNnz());
		out.setFedMapping(fedMap.copyWithNewID(fr1[0].getID()));
	}

	public void processFedInstruction(ExecutionContext ec, MatrixObject mo1, MatrixObject mo2, boolean reversed) {
		MatrixObject mo3 = input3 != null && input3.isMatrix() ? ec.getMatrixObject(input3) : null;
		Future<FederatedResponse>[] ffr;

		boolean reversedWeights = mo3 != null && mo3.isFederated() && !(mo1.isFederated() || mo2.isFederated());

		if(reversedWeights) {
			mo3 = mo1;
			mo1 = ec.getMatrixObject(input3);
		}

		FederatedRequest[] fr1 = mo1.getFedMapping().broadcastSliced(mo2, false);
		if(mo3 == null) {
			FederatedRequest fr2;
			if(!reversed)
				fr2 = FederationUtils.callInstruction(instString, output, new CPOperand[] {input1, input2}, new long[] {mo1.getFedMapping().getID(), fr1[0].getID()});
			else
				fr2 = FederationUtils.callInstruction(instString, output, new CPOperand[] {input1, input2}, new long[] {fr1[0].getID(), mo1.getFedMapping().getID()});

			FederatedRequest fr3 = new FederatedRequest(FederatedRequest.RequestType.GET_VAR, fr2.getID());
			FederatedRequest fr4 = mo1.getFedMapping().cleanup(getTID(), fr1[0].getID());
			ffr = mo1.getFedMapping().execute(getTID(), true, fr1, fr2, fr3, fr4);
		} else {
			FederatedRequest[] fr2 = mo1.getFedMapping().broadcastSliced(mo3, false);
			FederatedRequest fr3;
			if(!reversed && !reversedWeights)
				fr3 = FederationUtils.callInstruction(instString, output, new CPOperand[] {input1, input2, input3}, new long[] {mo1.getFedMapping().getID(), fr1[0].getID(), fr2[0].getID()});
			else if(reversed && !reversedWeights)
				fr3 = FederationUtils.callInstruction(instString, output, new CPOperand[] {input1, input2, input3}, new long[] {fr1[0].getID(), mo1.getFedMapping().getID(), fr2[0].getID()});
			else
				fr3 = FederationUtils.callInstruction(instString, output, new CPOperand[] {input1, input2, input3}, new long[] {fr1[0].getID(), fr2[0].getID(), mo1.getFedMapping().getID()});

			FederatedRequest fr4 = new FederatedRequest(FederatedRequest.RequestType.GET_VAR, fr3.getID());
			FederatedRequest fr5 = mo1.getFedMapping().cleanup(getTID(), fr1[0].getID(), fr2[0].getID());
			ffr = mo1.getFedMapping().execute(getTID(), true, fr1, fr2, fr3, fr4, fr5);
		}

		ec.setMatrixOutput(output.getName(), aggResult(ffr));
	}

	private MatrixBlock aggResult(Future<FederatedResponse>[] ffr) {
		MatrixBlock resultBlock = new MatrixBlock(1, 1, 0);
		int dim1 = 0, dim2 = 0;
		for(int i = 0; i < ffr.length; i++) {
			try {
				MatrixBlock mb = ((MatrixBlock) ffr[i].get().getData()[0]);
				dim1 = mb.getNumRows()  > dim1 ? mb.getNumRows() : dim1;
				dim2 = mb.getNumColumns()  > dim2 ? mb.getNumColumns() : dim2;

				// set next and prev to same output dimensions
				MatrixBlock prev = new MatrixBlock(dim1, dim2, 0.0);
				prev.copy(0, resultBlock.getNumRows()-1, 0, resultBlock.getNumColumns()-1, resultBlock, true);

				MatrixBlock next = new MatrixBlock(dim1, dim2, 0.0);
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

	private long[] getOutputDimension(ExecutionContext ec, CPOperand inOp, CPOperand outOp, FederatedRange[] federatedRanges) {
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

		// get max per worker
		FederationMap map = in.getFedMapping();
		FederatedRequest fr1 = FederationUtils.callInstruction(maxInstrString, outOp,
			new CPOperand[]{inOp}, new long[]{in.getFedMapping().getID()});
		FederatedRequest fr2 = new FederatedRequest(FederatedRequest.RequestType.GET_VAR, fr1.getID());
		FederatedRequest fr3 = map.cleanup(getTID(), fr1.getID());
		Future<FederatedResponse>[] tmp = map.execute(getTID(), fr1, fr2, fr3);

		// get result
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
}