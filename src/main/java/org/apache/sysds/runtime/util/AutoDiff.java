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

package org.apache.sysds.runtime.util;

import org.apache.commons.lang3.mutable.MutableInt;
import org.apache.sysds.common.Types;
import org.apache.sysds.hops.*;
import org.apache.sysds.hops.recompile.Recompiler;
import org.apache.sysds.hops.rewrite.HopRewriteUtils;
import org.apache.sysds.parser.DataExpression;
import org.apache.sysds.parser.DataIdentifier;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.BasicProgramBlock;
import org.apache.sysds.runtime.controlprogram.Program;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.instructions.InstructionParser;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.*;
import org.apache.sysds.runtime.instructions.spark.RandSPInstruction;
import org.apache.sysds.runtime.instructions.spark.SPInstruction;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.lineage.LineageParser;
import org.apache.sysds.runtime.lineage.LineageRecomputeUtils;

import java.util.*;

public class AutoDiff {
	private static final String ADVARPREFIX = "adVar";

	private static List<Data> getAffineBackward( MatrixObject dout, MatrixObject X, MatrixObject W,
		ExecutionContext adec) {

		final String AD_VARDX = "__adx";
		final String AD_VARDW = "__adw";
		final String AD_VARDB = "__adb";

		adec.setVariable("X", X);
		adec.setVariable("dout", dout);
		adec.setVariable("W", W);
		DataOp input = HopRewriteUtils.createTransientRead("X", X);
		DataOp output = HopRewriteUtils.createTransientRead("dout", dout);
		DataOp w = HopRewriteUtils.createTransientRead("W", W);

		//dx = dout %*% t(w1)
		ReorgOp trasnW = HopRewriteUtils.createTranspose(w);
		AggBinaryOp matMulX = HopRewriteUtils.createMatrixMultiply(output, trasnW);
		DataOp dX = HopRewriteUtils.createTransientWrite(AD_VARDX, matMulX);
		//dw = t(X) %*% dout
		ReorgOp trasnX = HopRewriteUtils.createTranspose(input);
		AggBinaryOp matMulW = HopRewriteUtils.createMatrixMultiply(trasnX, output);
		DataOp dW = HopRewriteUtils.createTransientWrite(AD_VARDW, matMulW);
		// db = colSums(bias)
		AggUnaryOp colSum = HopRewriteUtils.createAggUnaryOp(output, Types.AggOp.SUM, Types.Direction.Col);
		DataOp dB = HopRewriteUtils.createTransientWrite(AD_VARDB, colSum);
		// create dX instruction
		ArrayList<Instruction> dXInst = Recompiler
			.recompileHopsDag(dX, adec.getVariables(), null, true, true, 0);
		// create dW instruction
		ArrayList<Instruction> dWInst = Recompiler
			.recompileHopsDag(dW, adec.getVariables(), null, true, true, 0);
		// create db instruction
		ArrayList<Instruction> dBInst = Recompiler
			.recompileHopsDag(dB, adec.getVariables(), null, true, true, 0);
		executeInst(dXInst, adec);
		executeInst(dWInst, adec);
		executeInst(dBInst, adec);

		List<Data> outputList = new ArrayList<>();
		outputList.add(adec.getVariable(AD_VARDX));
		outputList.add(adec.getVariable(AD_VARDW));
		outputList.add(adec.getVariable(AD_VARDB));

		return outputList;
	}

	private static void executeInst(ArrayList<Instruction> newInst, ExecutionContext lrwec)
	{
		try {
			//execute instructions
			BasicProgramBlock pb = new BasicProgramBlock(new Program());
			pb.setInstructions(newInst);
			pb.execute(lrwec);
		}
		catch (Exception e) {
			throw new DMLRuntimeException("Error executing autoDiff instruction" , e);
		}
	}

	public static ListObject getBackward(ArrayList<Data> forwardLayers, ArrayList<Data> lineage, ExecutionContext adec) {
		List<Data> data = new ArrayList<Data>();
		List<String> names = new ArrayList<String>();
		// reverse the list to start from the last layer
		Collections.reverse(forwardLayers);
		Collections.reverse(lineage);
		for(int i=0; i<forwardLayers.size(); i++)
		{
			// get the name
			String layer = forwardLayers.get(i).toString();
			if(layer.equals("affine"))
			{
				// get the lineage
				String lin = lineage.get(i).toString();
				// get rid of foo flag
				lin = lin.replace("foo", "");

				// get lineage of X, Y, W and dout for affine backward
				String[] linInst = lin.split("\n");
				// assumption first line in the lineage is X and second is weight
				MatrixObject dout = (MatrixObject)LineageRecomputeUtils.parseNComputeLineageTrace(lin, null);
				MatrixObject X =  (MatrixObject)LineageRecomputeUtils.parseNComputeLineageTrace(linInst[0], null);
				MatrixObject W =  (MatrixObject)LineageRecomputeUtils.parseNComputeLineageTrace(linInst[1], null);
				// prepare the named list
				names.add("dX");
				names.add("dW");
				names.add("dB");
				data = getAffineBackward(dout, X, W, adec);
			}
		}
		return new ListObject(data, names);
	}
	public static ListObject getBackward(MatrixObject mo, ArrayList<Data> lineage, ExecutionContext adec) {

		ArrayList<String> names = new ArrayList<String>();
		// parse the lineage and take the number of instructions as for each instruction there is separate hop DAG
		String lin = lineage.get(0).toString();
		// get rid of foo flag
		lin = lin.replace("foo", "");
		List<Data>  data = parseNComputeAutoDiffFromLineage(mo, lin, names, adec);
		return new ListObject(data, names);
	}

	public static List<Data> parseNComputeAutoDiffFromLineage(MatrixObject mo, String mainTrace,
		ArrayList<String> names, ExecutionContext ec ) {

		LineageItem root = LineageParser.parseLineageTrace(mainTrace);
		// Recursively construct hops
		root.resetVisitStatusNR();
		Map<Long, Hop> operands = new HashMap<>();
		// set variable for input matrix
		ec.setVariable("X", mo);
		DataOp input = HopRewriteUtils.createTransientRead("X", mo);
		// each instruction Hop is stored separately as each instruction creates a new differentiation
		ArrayList<Hop> allHops = constructHopsNR(root, operands, input, names);

		ArrayList<Data> results = new ArrayList<>();
		for(int i=0; i< allHops.size(); i++) {
			DataOp dop = HopRewriteUtils.createTransientWrite("advar"+i, allHops.get(i));
			ArrayList<Instruction> dInst = Recompiler
				.recompileHopsDag(dop, ec.getVariables(), null, true, true, 0);
			// create derivative instructions
			executeInst(dInst, ec);
			results.add(ec.getVariable("advar"+i));
		}
		return results;
	}


	public static ArrayList<Hop> constructHopsNR(LineageItem item, Map<Long, Hop> operands,	Hop mo, ArrayList<String> names)
	{

		//skeleton as explainLineageItemNR
		ArrayList<Hop>  allHops = new ArrayList<>();
		Stack<LineageItem> stackItem = new Stack<>();
		Stack<MutableInt> stackPos = new Stack<>();
		stackItem.push(item); stackPos.push(new MutableInt(0));
		while (!stackItem.empty()) {
			LineageItem tmpItem = stackItem.peek();
			MutableInt tmpPos = stackPos.peek();
			//check ascent condition - no item processing
			if (tmpItem.isVisited()) {
				stackItem.pop(); stackPos.pop();
			}
			//check ascent condition - append item
			else if( tmpItem.getInputs() == null
				|| tmpItem.getInputs().length <= tmpPos.intValue() ) {
				constructSingleHop(tmpItem, operands, mo, allHops, names);
				stackItem.pop(); stackPos.pop();
				tmpItem.setVisited();
			}
			//check descent condition
			else if( tmpItem.getInputs() != null ) {
				stackItem.push(tmpItem.getInputs()[tmpPos.intValue()]);
				tmpPos.increment();
				stackPos.push(new MutableInt(0));
			}
		}
		return allHops;
	}

	private static void constructSingleHop(LineageItem item, Map<Long, Hop> operands, Hop mo, ArrayList<Hop> allHops, ArrayList<String> names)
	{
		//process current lineage item
		switch (item.getType()) {
			case Creation: {
				if(item.getData().startsWith(ADVARPREFIX)) {
					long phId = Long.parseLong(item.getData().substring(3));
					Hop input = operands.get(phId);
					operands.remove(phId);
					// Replace the placeholders with TReads
					operands.put(item.getId(), input); // order preserving
					break;
				}
				Instruction inst = InstructionParser.parseSingleInstruction(item.getData());

				if(inst instanceof DataGenCPInstruction) {
					DataGenCPInstruction rand = (DataGenCPInstruction) inst;
					HashMap<String, Hop> params = new HashMap<>();
					if(rand.getOpcode().equals("rand")) {
						if(rand.output.getDataType() == Types.DataType.TENSOR)
							params.put(DataExpression.RAND_DIMS, new LiteralOp(rand.getDims()));
						else {
							params.put(DataExpression.RAND_ROWS, new LiteralOp(rand.getRows()));
							params.put(DataExpression.RAND_COLS, new LiteralOp(rand.getCols()));
						}
						params.put(DataExpression.RAND_MIN, new LiteralOp(rand.getMinValue()));
						params.put(DataExpression.RAND_MAX, new LiteralOp(rand.getMaxValue()));
						params.put(DataExpression.RAND_PDF, new LiteralOp(rand.getPdf()));
						params.put(DataExpression.RAND_LAMBDA, new LiteralOp(rand.getPdfParams()));
						params.put(DataExpression.RAND_SPARSITY, new LiteralOp(rand.getSparsity()));
						params.put(DataExpression.RAND_SEED, new LiteralOp(rand.getSeed()));
					}
					Hop datagen = new DataGenOp(Types.OpOpDG.valueOf(rand.getOpcode().toUpperCase()),
						new DataIdentifier("tmp"), params);
					datagen.setBlocksize(rand.getBlocksize());
					operands.put(item.getId(), datagen);
				}
				else if(inst instanceof VariableCPInstruction && ((VariableCPInstruction) inst).isCreateVariable()) {
					String parts[] = InstructionUtils.getInstructionPartsWithValueType(inst.toString());
					Types.DataType dt = Types.DataType.valueOf(parts[4]);
					Types.ValueType vt = dt == Types.DataType.MATRIX ? Types.ValueType.FP64 : Types.ValueType.STRING;
					HashMap<String, Hop> params = new HashMap<>();
					params.put(DataExpression.IO_FILENAME, new LiteralOp(parts[2]));
					params.put(DataExpression.READROWPARAM, new LiteralOp(Long.parseLong(parts[6])));
					params.put(DataExpression.READCOLPARAM, new LiteralOp(Long.parseLong(parts[7])));
					params.put(DataExpression.READNNZPARAM, new LiteralOp(Long.parseLong(parts[8])));
					params.put(DataExpression.FORMAT_TYPE, new LiteralOp(parts[5]));
					DataOp pread = new DataOp(parts[1].substring(5), dt, vt, Types.OpOpData.PERSISTENTREAD, params);
					pread.setFileName(parts[2]);
					operands.put(item.getId(), pread);
				}
				else if(inst instanceof RandSPInstruction) {
					RandSPInstruction rand = (RandSPInstruction) inst;
					HashMap<String, Hop> params = new HashMap<>();
					if(rand.output.getDataType() == Types.DataType.TENSOR)
						params.put(DataExpression.RAND_DIMS, new LiteralOp(rand.getDims()));
					else {
						params.put(DataExpression.RAND_ROWS, new LiteralOp(rand.getRows()));
						params.put(DataExpression.RAND_COLS, new LiteralOp(rand.getCols()));
					}
					params.put(DataExpression.RAND_MIN, new LiteralOp(rand.getMinValue()));
					params.put(DataExpression.RAND_MAX, new LiteralOp(rand.getMaxValue()));
					params.put(DataExpression.RAND_PDF, new LiteralOp(rand.getPdf()));
					params.put(DataExpression.RAND_LAMBDA, new LiteralOp(rand.getPdfParams()));
					params.put(DataExpression.RAND_SPARSITY, new LiteralOp(rand.getSparsity()));
					params.put(DataExpression.RAND_SEED, new LiteralOp(rand.getSeed()));
					Hop datagen = new DataGenOp(Types.OpOpDG.RAND, new DataIdentifier("tmp"), params);
					datagen.setBlocksize(rand.getBlocksize());
					operands.put(item.getId(), datagen);
				}
				break;
			}
			case Instruction: {
				CPInstruction.CPType ctype = InstructionUtils.getCPTypeByOpcode(item.getOpcode());
				SPInstruction.SPType stype = InstructionUtils.getSPTypeByOpcode(item.getOpcode());

				if(ctype != null) {
					switch(ctype) {
						case AggregateBinary: {
							Hop input1 = operands.get(item.getInputs()[0].getId());
							Hop input2 = operands.get(item.getInputs()[1].getId());
							ReorgOp trasnX = HopRewriteUtils.createTranspose(input1);
							ReorgOp trasnW = HopRewriteUtils.createTranspose(input2);
							Hop dX = HopRewriteUtils.createMatrixMultiply(mo, trasnW);
							Hop dW = HopRewriteUtils.createMatrixMultiply(trasnX, mo);
							operands.put(item.getId(), dX);
							operands.put(item.getId() + 1, dW);
							allHops.add(dX);
							allHops.add(dW);
							names.add("dX");
							names.add("dW");
							break;
						}
						case Binary: {
							//handle special cases of binary operations
							String opcode = item.getOpcode();
							Hop output = null;
							if(opcode.equals("+"))
								output = HopRewriteUtils.createAggUnaryOp(mo, Types.AggOp.SUM, Types.Direction.Col);
							operands.put(item.getId(), output);
							allHops.add(output);
							names.add("dB");
							break;
						}
						default:
							throw new DMLRuntimeException(
								"Unsupported autoDiff instruction " + "type: " + ctype.name() + " (" + item.getOpcode() + ").");
					}
				}
			}
		}
	}
}
