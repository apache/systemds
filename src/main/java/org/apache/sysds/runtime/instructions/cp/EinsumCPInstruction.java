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

package org.apache.sysds.runtime.instructions.cp;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.lang3.tuple.Triple;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.hops.LiteralOp;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.hops.codegen.SpoofCompiler;
import org.apache.sysds.hops.codegen.cplan.CNode;
import org.apache.sysds.hops.codegen.cplan.CNodeBinary;
import org.apache.sysds.hops.codegen.cplan.CNodeCell;
import org.apache.sysds.hops.codegen.cplan.CNodeData;
import org.apache.sysds.hops.codegen.cplan.CNodeNary;
import org.apache.sysds.hops.codegen.cplan.CNodeRow;
import org.apache.sysds.runtime.codegen.*;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.einsum.EinsumContext;
import org.apache.sysds.runtime.functionobjects.*;
import org.apache.sysds.runtime.matrix.data.LibMatrixMult;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.AggregateOperator;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.matrix.operators.ReorgOperator;
import org.apache.sysds.runtime.matrix.operators.SimpleOperator;

import java.util.*;
import java.util.function.Predicate;

public class EinsumCPInstruction extends BuiltinNaryCPInstruction {
	public static boolean FORCE_CELL_TPL = false;
	protected static final Log LOG = LogFactory.getLog(EinsumCPInstruction.class.getName());
	public String eqStr;
	private final int _numThreads;
	private final CPOperand[] _in;

	public EinsumCPInstruction(Operator op, String opcode, String istr, CPOperand out, CPOperand... inputs)
	{
		super(op, opcode, istr, out, inputs);
		_numThreads = OptimizerUtils.getConstrainedNumThreads(-1);
		_in = inputs;
		this.eqStr = inputs[0].getName();
		Logger.getLogger(EinsumCPInstruction.class).setLevel(Level.TRACE);
	}

	private EinsumContext einc = null;

	@Override
	public void processInstruction(ExecutionContext ec) {
		//get input matrices and scalars, incl pinning of matrices
		ArrayList<MatrixBlock> inputs = new ArrayList<>();
		for (CPOperand input : _in) {
			if(input.getDataType()==DataType.MATRIX){
				MatrixBlock mb = ec.getMatrixInput(input.getName());
				if(mb instanceof CompressedMatrixBlock){
					mb = ((CompressedMatrixBlock) mb).getUncompressed("Spoof instruction");
				}
				inputs.add(mb);
			}
		}

		EinsumContext einc = EinsumContext.getEinsumContext(eqStr, inputs);

		this.einc = einc;
		String resultString = einc.outChar2 != null ? String.valueOf(einc.outChar1) + einc.outChar2 : einc.outChar1 != null ? String.valueOf(einc.outChar1) : "";

		if( LOG.isDebugEnabled() ) LOG.trace("outrows:"+einc.outRows+", outcols:"+einc.outCols);

		ArrayList<String> inputsChars = einc.newEquationStringInputsSplit;

		if(LOG.isTraceEnabled()) LOG.trace(String.join(",",einc.newEquationStringInputsSplit));

		contractDimensionsAndComputeDiagonals(einc, inputs);

		//make all vetors col vectors
		for(int i = 0; i < inputs.size(); i++){
			if(inputs.get(i) != null && inputsChars.get(i).length() == 1) EnsureMatrixBlockColumnVector(inputs.get(i));
		}

		if(LOG.isTraceEnabled()) for(Character c : einc.characterAppearanceIndexes.keySet()){
			ArrayList<Integer> a = einc.characterAppearanceIndexes.get(c);
			LOG.trace(c+" count= "+a.size());
		}

		// compute scalar by suming-all matrices:
		Double scalar = null;
		for(int i=0;i< inputs.size(); i++){
			String s = inputsChars.get(i);
			if(s.equals("")){
				MatrixBlock mb = inputs.get(i);
				if (scalar == null) scalar = mb.get(0,0);
				else scalar*= mb.get(0,0);
				inputs.set(i,null);
				inputsChars.set(i,null);
			}
		}

		if (scalar != null) {
			inputsChars.add("");
			inputs.add(new MatrixBlock(scalar));
		}

		HashMap<Character, Integer> characterToOccurences = new HashMap<>();
		for (Character key :einc.characterAppearanceIndexes.keySet()) {
			characterToOccurences.put(key, einc.characterAppearanceIndexes.get(key).size());
		}
		for (Character key :einc.charToDimensionSize.keySet()) {
			if(!characterToOccurences.containsKey(key))
				characterToOccurences.put(key, 1);
		}

		ArrayList<EOpNode> eOpNodes = new ArrayList<>(inputsChars.size());
		for (int i = 0; i < inputsChars.size(); i++) {
			if (inputsChars.get(i) == null) continue;
			EOpNodeData n = new EOpNodeData(inputsChars.get(i).length() > 0 ? inputsChars.get(i).charAt(0) : null, inputsChars.get(i).length() > 1 ? inputsChars.get(i).charAt(1) : null, i);
			eOpNodes.add(n);
		}
		Pair<Integer, List<EOpNode> > plan = FORCE_CELL_TPL ? null : generatePlan(0, eOpNodes, einc.charToDimensionSize, characterToOccurences, einc.outChar1, einc.outChar2);


		ArrayList<MatrixBlock> resMatrices = FORCE_CELL_TPL ? null : executePlan(plan.getRight(), inputs);
//		ArrayList<MatrixBlock> resMatrices = executePlan(plan.getRight(), inputs, true);

		if(!FORCE_CELL_TPL && resMatrices.size() == 1){
			EOpNode resNode = plan.getRight().get(0);
			if (einc.outChar1 != null && einc.outChar2 != null){
				if(resNode.c1 == einc.outChar1 && resNode.c2 == einc.outChar2){
					ec.setMatrixOutput(output.getName(), resMatrices.get(0));
				}
				else if(resNode.c1 == einc.outChar2 && resNode.c2 == einc.outChar1){
					ReorgOperator transpose = new ReorgOperator(SwapIndex.getSwapIndexFnObject(), _numThreads);
					MatrixBlock resM = resMatrices.get(0).reorgOperations(transpose, new MatrixBlock(),0,0,0);
					ec.setMatrixOutput(output.getName(), resM);
				}else{
					if(LOG.isTraceEnabled()) LOG.trace("Einsum expected: "+resultString + ", got: "+resNode.c1+resNode.c2);
					throw new RuntimeException("Einsum plan produced different result");
				}
			}else if (einc.outChar1 != null){
				if(resNode.c1 == einc.outChar1  && resNode.c2 == null){
					ec.setMatrixOutput(output.getName(), resMatrices.get(0));
				}else{
					if(LOG.isTraceEnabled()) LOG.trace("Einsum expected: "+resultString + ", got: "+resNode.c1+resNode.c2);
					throw new RuntimeException("Einsum plan produced different result");
				}
			}else{
				if(resNode.c1 == null && resNode.c2 == null){
					ec.setScalarOutput(output.getName(), new DoubleObject(resMatrices.get(0).get(0, 0)));;
				}
			}
		}else{
			// use cell template with loops for remaining
			ArrayList<MatrixBlock> mbs = resMatrices;
			ArrayList<String> chars = new ArrayList<>();

			for (int i = 0; i < plan.getRight().size(); i++) {
				String s;
				if(plan.getRight().get(i).c1 == null) s = "";
				else if(plan.getRight().get(i).c2 == null) s = plan.getRight().get(i).c1.toString();
				else s = plan.getRight().get(i).c1.toString() + plan.getRight().get(i).c2;
				chars.add(s);
			}

			ArrayList<Character> summingChars = new ArrayList<>();
			for (Character c : einc.characterAppearanceIndexes.keySet()) {
				if (c != einc.outChar1 && c != einc.outChar2) summingChars.add(c);
			}
			if(LOG.isTraceEnabled()) LOG.trace("finishing with cell tpl: "+String.join(",", chars));

			MatrixBlock res = computeCellSummation(mbs, chars, resultString, einc.charToDimensionSize, summingChars, einc.outRows, einc.outCols);

			if (einc.outRows == 1 && einc.outCols == 1)
				ec.setScalarOutput(output.getName(), new DoubleObject(res.get(0, 0)));
			else ec.setMatrixOutput(output.getName(), res);
		}
		if(LOG.isTraceEnabled()) LOG.trace("EinsumCPInstruction Finished");

		releaseMatrixInputs(ec);

	}

	private void contractDimensionsAndComputeDiagonals(EinsumContext einc, ArrayList<MatrixBlock> inputs) {
		for(int i = 0; i< einc.contractDims.length; i++){
			//AggregateOperator agg = new AggregateOperator(0, KahanPlus.getKahanPlusFnObject(),Types.CorrectionLocationType.LASTCOLUMN);
			AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());

			if(einc.diagonalInputs[i]){
				ReorgOperator op = new ReorgOperator(DiagIndex.getDiagIndexFnObject());
				inputs.set(i, inputs.get(i).reorgOperations(op, new MatrixBlock(),0,0,0));
			}
			if (einc.contractDims[i] == null) continue;
			switch (einc.contractDims[i]){
				case CONTRACT_BOTH: {
					AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceAll.getReduceAllFnObject(), _numThreads);
					MatrixBlock res = new MatrixBlock(1, 1, false);
					inputs.get(i).aggregateUnaryOperations(aggun, res, 0, null);
					inputs.set(i, res);
					break;
				}
				case CONTRACT_RIGHT: {
					AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceCol.getReduceColFnObject(), _numThreads);
					MatrixBlock res = new MatrixBlock(inputs.get(i).getNumRows(), 1, false);
					inputs.get(i).aggregateUnaryOperations(aggun, res, 0, null);
					inputs.set(i, res);
					break;
				}
				case CONTRACT_LEFT: {
					AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceRow.getReduceRowFnObject(), _numThreads);
					MatrixBlock res = new MatrixBlock(inputs.get(i).getNumColumns(), 1, false);
					inputs.get(i).aggregateUnaryOperations(aggun, res, 0, null);
					inputs.set(i, res);
					break;
				}
				default:
					break;
			}
		}
	}

	private enum EBinaryOperand { // upper case: char has to remain, lower case: to be summed
		////// summations:   //////
		aB_a,// -> B
		Ba_a, // -> B
		Ba_aC, // mmult -> BC
		aB_Ca,
		Ba_Ca, // -> BC
		aB_aC, // outer mult, possibly with transposing first -> BC
		a_a,// dot ->

		////// elementwisemult and sums, something like ij,ij->i   //////
		aB_aB,// elemwise and colsum -> B
		Ba_Ba, // elemwise and rowsum ->B
		Ba_aB, // elemwise, either colsum or rowsum -> B
//		aB_Ba,

		////// elementwise, no summations:   //////
		A_A,// v-elemwise -> A
		AB_AB,// M-M elemwise -> AB
		AB_BA, // M-M.T elemwise -> AB
		AB_A, // M-v colwise -> BA!?
		BA_A, // M-v rowwise -> BA
		ab_ab,//M-M sum all
		ab_ba, //M-M.T sum all
		////// other   //////
		A_B, // outer mult -> AB
		A_scalar, // v-scalar
		AB_scalar, // m-scalar
		scalar_scalar
	}
	private abstract class EOpNode {
		public Character c1;
		public Character c2; // nullable
		public EOpNode(Character c1, Character c2){
			this.c1 = c1;
			this.c2 = c2;
		}
	}
	private class EOpNodeBinary extends EOpNode {
		public EOpNode left;
		public EOpNode right;
		public EBinaryOperand operand;
		public EOpNodeBinary(Character c1, Character c2, EOpNode left, EOpNode right, EBinaryOperand operand){
			super(c1,c2);
			this.left = left;
			this.right = right;
			this.operand = operand;
		}
	}
	private class EOpNodeData extends EOpNode {
		public int matrixIdx;
		public EOpNodeData(Character c1, Character c2, int matrixIdx){
			super(c1,c2);
			this.matrixIdx = matrixIdx;
		}
	}

	private Pair<Integer, List<EOpNode> /* ideally with one element */> generatePlan(int cost, ArrayList<EOpNode> operands, HashMap<Character, Integer> charToSizeMap, HashMap<Character, Integer> charToOccurences, Character outChar1, Character outChar2) {
		Integer minCost = cost;
		List<EOpNode> minNodes = operands;

		if (operands.size() == 2){
			boolean swap = (operands.get(0).c2 == null && operands.get(1).c2 != null) || operands.get(0).c1 == null;
			EOpNode n1 = operands.get(!swap ? 0 : 1);
			EOpNode n2 = operands.get(!swap ? 1 : 0);
			Triple<Integer, EBinaryOperand, Pair<Character, Character>> t = TryCombineAndCost(n1, n2, charToSizeMap, charToOccurences, outChar1, outChar2);
			if (t != null) {
				EOpNodeBinary newNode = new EOpNodeBinary(t.getRight().getLeft(), t.getRight().getRight(), n1, n2, t.getMiddle());
				int thisCost = cost + t.getLeft();
				return Pair.of(thisCost, Arrays.asList(newNode));
			}
			return Pair.of(cost, operands);
		}
		else if (operands.size() == 1){
			// check for transpose
			return Pair.of(cost, operands);
		}

		for(int i = 0; i < operands.size()-1; i++){
			for (int j = i+1; j < operands.size(); j++){
				boolean swap = (operands.get(i).c2 == null && operands.get(j).c2 != null) || operands.get(i).c1 == null;
				EOpNode n1 = operands.get(!swap ? i : j);
				EOpNode n2 = operands.get(!swap ? j : i);


				Triple<Integer, EBinaryOperand, Pair<Character, Character>> t = TryCombineAndCost(n1, n2, charToSizeMap, charToOccurences, outChar1, outChar2);
				if (t != null){
					EOpNodeBinary newNode = new EOpNodeBinary(t.getRight().getLeft(), t.getRight().getRight(), n1, n2, t.getMiddle());
					int thisCost = cost + t.getLeft();

					if(n1.c1 != null) charToOccurences.put(n1.c1, charToOccurences.get(n1.c1)-1);
					if(n1.c2 != null) charToOccurences.put(n1.c2, charToOccurences.get(n1.c2)-1);
					if(n2.c1 != null) charToOccurences.put(n2.c1, charToOccurences.get(n2.c1)-1);
					if(n2.c2 != null) charToOccurences.put(n2.c2, charToOccurences.get(n2.c2)-1);

					if(newNode.c1 != null) charToOccurences.put(newNode.c1, charToOccurences.get(newNode.c1)+1);
					if(newNode.c2 != null) charToOccurences.put(newNode.c2, charToOccurences.get(newNode.c2)+1);

					ArrayList<EOpNode> newOperands = new ArrayList<>(operands.size()-1);
					for(int z = 0; z < operands.size(); z++){
						if(z != i && z != j) newOperands.add(operands.get(z));
					}
					newOperands.add(newNode);

					Pair<Integer, List<EOpNode>> furtherPlan = generatePlan(thisCost, newOperands,charToSizeMap, charToOccurences, outChar1, outChar2);
					if(furtherPlan.getRight().size() < (minNodes.size()) || furtherPlan.getLeft() < minCost){
						minCost = furtherPlan.getLeft();
						minNodes = furtherPlan.getRight();
					}

					if(n1.c1 != null) charToOccurences.put(n1.c1, charToOccurences.get(n1.c1)+1);
					if(n1.c2 != null) charToOccurences.put(n1.c2, charToOccurences.get(n1.c2)+1);
					if(n2.c1 != null) charToOccurences.put(n2.c1, charToOccurences.get(n2.c1)+1);
					if(n2.c2 != null) charToOccurences.put(n2.c2, charToOccurences.get(n2.c2)+1);
					if(newNode.c1 != null) charToOccurences.put(newNode.c1, charToOccurences.get(newNode.c1)-1);
					if(newNode.c2 != null) charToOccurences.put(newNode.c2, charToOccurences.get(newNode.c2)-1);
				}
			}
		}

		return Pair.of(minCost, minNodes);
	}

	private static Triple<Integer, EBinaryOperand, Pair<Character, Character>> TryCombineAndCost(EOpNode n1 , EOpNode n2, HashMap<Character, Integer> charToSizeMap, HashMap<Character, Integer> charToOccurences, Character outChar1, Character outChar2){
		Predicate<Character> cannotBeSummed = (c) ->
				c == outChar1 || c == outChar2 || charToOccurences.get(c) > 2;

		if(n1.c1 == null) {
			// n2.c1 also has to be null
			return Triple.of(1, EBinaryOperand.scalar_scalar, Pair.of(null, null));
		}

		if(n2.c1 == null) {
			if(n1.c2 == null)
				return Triple.of(charToSizeMap.get(n1.c1), EBinaryOperand.A_scalar, Pair.of(n1.c1, null));
			return Triple.of(charToSizeMap.get(n1.c1)*charToSizeMap.get(n1.c2), EBinaryOperand.AB_scalar, Pair.of(n1.c1, n1.c2));
		}

		if(n1.c1 == n2.c1){
			if(n1.c2 != null){
				if ( n1.c2 == n2.c2){
					if( cannotBeSummed.test(n1.c1)){
						if(cannotBeSummed.test(n1.c2)){
							return Triple.of(charToSizeMap.get(n1.c1)*charToSizeMap.get(n1.c2), EBinaryOperand.AB_AB, Pair.of(n1.c1, n1.c2));
						}
						return Triple.of(charToSizeMap.get(n1.c1)*charToSizeMap.get(n1.c2), EBinaryOperand.Ba_Ba, Pair.of(n1.c1, null));
					}

					if(cannotBeSummed.test(n1.c2)){
						return Triple.of(charToSizeMap.get(n1.c1)*charToSizeMap.get(n1.c2), EBinaryOperand.aB_aB, Pair.of(n1.c2, null));
					}

					return Triple.of(charToSizeMap.get(n1.c1)*charToSizeMap.get(n1.c2), EBinaryOperand.ab_ab, Pair.of(null, null));

				}

				else if(n2.c2 == null){
					if(cannotBeSummed.test(n1.c1)){
						return Triple.of(charToSizeMap.get(n1.c1)*charToSizeMap.get(n1.c2)*2, EBinaryOperand.AB_A, Pair.of(n1.c1, n1.c2));
					}
					return Triple.of(charToSizeMap.get(n1.c1)*charToSizeMap.get(n1.c2)*2, EBinaryOperand.aB_a, Pair.of(n1.c2, null)); // in theory (null, n1.c2)
				}
				else if(n1.c1 ==outChar1 || n1.c1==outChar2|| charToOccurences.get(n1.c1) > 2){
					return null;// AB,AC
				}
				else {
					return Triple.of((charToSizeMap.get(n1.c1)*charToSizeMap.get(n1.c2))+(charToSizeMap.get(n1.c1)*charToSizeMap.get(n1.c2)*charToSizeMap.get(n2.c2)), EBinaryOperand.aB_aC, Pair.of(n1.c2, n2.c2)); // or n2.c2, n1.c2
				}
			}else{ // n1.c2 = null -> c2.c2 = null
				if(n1.c1 ==outChar1 || n1.c1==outChar2 || charToOccurences.get(n1.c1) > 2){
					return Triple.of(charToSizeMap.get(n1.c1), EBinaryOperand.A_A, Pair.of(n1.c1, null));
				}
				return Triple.of(charToSizeMap.get(n1.c1), EBinaryOperand.a_a, Pair.of(null, null));
			}


		}else{ // n1.c1 != n2.c1
			if(n1.c2 == null) {
				return Triple.of(charToSizeMap.get(n1.c1)*charToSizeMap.get(n2.c1), EBinaryOperand.A_B, Pair.of(n1.c1, n2.c1));
			}
			else if(n2.c2 == null) { // ab,c
				if (n1.c2 == n2.c1) {
					if(cannotBeSummed.test(n1.c2)){
						return Triple.of(charToSizeMap.get(n1.c1)*charToSizeMap.get(n2.c1), EBinaryOperand.BA_A, Pair.of(n1.c1, n1.c2));
					}
					return Triple.of(charToSizeMap.get(n1.c1)*charToSizeMap.get(n2.c1), EBinaryOperand.Ba_a, Pair.of(n1.c1, null));
				}
				return null; // AB,C
			}
			else if (n1.c2 == n2.c1) {
				if(n1.c1 == n2.c2){ // ab,ba
					if(cannotBeSummed.test(n1.c1)){
						if(cannotBeSummed.test(n1.c2)){
							return Triple.of(charToSizeMap.get(n1.c1)*charToSizeMap.get(n1.c2), EBinaryOperand.AB_BA, Pair.of(n1.c1, n1.c2));
						}
						return Triple.of(charToSizeMap.get(n1.c1)*charToSizeMap.get(n1.c2), EBinaryOperand.Ba_aB, Pair.of(n1.c1, null));
					}
					return Triple.of(charToSizeMap.get(n1.c1)*charToSizeMap.get(n1.c2), EBinaryOperand.ab_ba, Pair.of(null, null));
				}
				if(cannotBeSummed.test(n1.c2)){
					return null; // AB_B
				}else{
					return Triple.of(charToSizeMap.get(n1.c1)*charToSizeMap.get(n1.c2)*charToSizeMap.get(n2.c2), EBinaryOperand.Ba_aC, Pair.of(n1.c1, n2.c2));
//					if(n1.c1 ==outChar1 || n1.c1==outChar2|| charToOccurences.get(n1.c1) > 2){
//						return null; // AB_B
//					}
//					return Triple.of(charToSizeMap.get(n1.c1)*charToSizeMap.get(n1.c2), EBinaryOperand.Ba_a, Pair.of(n1.c1, null));
				}
			}
			if(n1.c1 == n2.c2) {
				if(cannotBeSummed.test(n1.c1)){
					return null; // AB_B
				}
				return Triple.of(charToSizeMap.get(n1.c1)*charToSizeMap.get(n1.c2)*charToSizeMap.get(n2.c1), EBinaryOperand.aB_Ca, Pair.of(n2.c1, n1.c2)); // * its just reorder of mmult
			}
			else if (n1.c2 == n2.c2) {
				if(n1.c2 ==outChar1 || n1.c2==outChar2|| charToOccurences.get(n1.c2) > 2){
					return null; // BA_CA
				}else{
					return Triple.of(charToSizeMap.get(n1.c1)*charToSizeMap.get(n1.c2) +(charToSizeMap.get(n1.c1)*charToSizeMap.get(n1.c2)*charToSizeMap.get(n2.c1)), EBinaryOperand.Ba_Ca, Pair.of(n1.c1, n2.c1)); // or n2.c1, n1.c1
				}
			}
			else { // we have something like ab,cd
				return null;
			}
		}
	}

	private ArrayList<MatrixBlock /* #els = #els of plan */> executePlan(List<EOpNode> plan, ArrayList<MatrixBlock> inputs){
		return executePlan(plan, inputs, false);
	}
	private ArrayList<MatrixBlock /* #els = #els of plan */> executePlan(List<EOpNode> plan, ArrayList<MatrixBlock> inputs, boolean codegen) {
		ArrayList<MatrixBlock> res = new ArrayList<>(plan.size());
		for(EOpNode p : plan){
			if(codegen) res.add(ComputeEOpNodeCodegen(p, inputs));
			else res.add(ComputeEOpNode(p, inputs));
		}
		return res;
	}

	private MatrixBlock ComputeEOpNode(EOpNode eOpNode, ArrayList<MatrixBlock> inputs){
		if(eOpNode instanceof EOpNodeData eOpNodeData){
			return inputs.get(eOpNodeData.matrixIdx);
		}
		EOpNodeBinary bin = (EOpNodeBinary) eOpNode;
		MatrixBlock left = ComputeEOpNode(bin.left, inputs);
		MatrixBlock right = ComputeEOpNode(bin.right, inputs);

		AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());

		MatrixBlock res;
		switch (bin.operand){
			case AB_AB -> {
				res = MatrixBlock.naryOperations(new SimpleOperator(Multiply.getMultiplyFnObject()), new MatrixBlock[]{left, right},new ScalarObject[]{}, new MatrixBlock());
			}
			case A_A -> {
				EnsureMatrixBlockColumnVector(left);
				EnsureMatrixBlockColumnVector(right);
				res = MatrixBlock.naryOperations(new SimpleOperator(Multiply.getMultiplyFnObject()), new MatrixBlock[]{left, right},new ScalarObject[]{}, new MatrixBlock());
			}
			case a_a -> {
				EnsureMatrixBlockColumnVector(left);
				EnsureMatrixBlockColumnVector(right);
				res = MatrixBlock.naryOperations(new SimpleOperator(Multiply.getMultiplyFnObject()), new MatrixBlock[]{left, right},new ScalarObject[]{}, new MatrixBlock());
				AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceAll.getReduceAllFnObject(), _numThreads);
				res = (MatrixBlock) res.aggregateUnaryOperations(aggun, new MatrixBlock(), 0, null);
			}
			////////////
			case Ba_Ba -> {
				res = MatrixBlock.naryOperations(new SimpleOperator(Multiply.getMultiplyFnObject()), new MatrixBlock[]{left, right},new ScalarObject[]{}, new MatrixBlock());
				AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceCol.getReduceColFnObject(), _numThreads);
				res = (MatrixBlock) res.aggregateUnaryOperations(aggun, new MatrixBlock(), 0, null);
			}
			case aB_aB -> {
				res = MatrixBlock.naryOperations(new SimpleOperator(Multiply.getMultiplyFnObject()), new MatrixBlock[]{left, right},new ScalarObject[]{}, new MatrixBlock());
				AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceRow.getReduceRowFnObject(), _numThreads);
				res = (MatrixBlock) res.aggregateUnaryOperations(aggun, new MatrixBlock(), 0, null);
				EnsureMatrixBlockColumnVector(res);
			}
			case ab_ab -> {
				res = MatrixBlock.naryOperations(new SimpleOperator(Multiply.getMultiplyFnObject()), new MatrixBlock[]{left, right},new ScalarObject[]{}, new MatrixBlock());
				AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceAll.getReduceAllFnObject(), _numThreads);
				res = (MatrixBlock) res.aggregateUnaryOperations(aggun, new MatrixBlock(), 0, null);
			}
			case ab_ba -> {
				ReorgOperator transpose = new ReorgOperator(SwapIndex.getSwapIndexFnObject(), _numThreads);
				right = right.reorgOperations(transpose, new MatrixBlock(), 0, 0, 0);
				res = MatrixBlock.naryOperations(new SimpleOperator(Multiply.getMultiplyFnObject()), new MatrixBlock[]{left, right},new ScalarObject[]{}, new MatrixBlock());
				AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceAll.getReduceAllFnObject(), _numThreads);
				res = (MatrixBlock) res.aggregateUnaryOperations(aggun, new MatrixBlock(), 0, null);
			}
			case Ba_aB -> {
				ReorgOperator transpose = new ReorgOperator(SwapIndex.getSwapIndexFnObject(), _numThreads);
				right = right.reorgOperations(transpose, new MatrixBlock(), 0, 0, 0);
				res = MatrixBlock.naryOperations(new SimpleOperator(Multiply.getMultiplyFnObject()), new MatrixBlock[]{left, right},new ScalarObject[]{}, new MatrixBlock());
				AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceCol.getReduceColFnObject(), _numThreads);
				res = (MatrixBlock) res.aggregateUnaryOperations(aggun, new MatrixBlock(), 0, null);
			}

			/////////
			case AB_BA -> {
				ReorgOperator transpose = new ReorgOperator(SwapIndex.getSwapIndexFnObject(), _numThreads);
				right = right.reorgOperations(transpose, new MatrixBlock(), 0, 0, 0);
				res = MatrixBlock.naryOperations(new SimpleOperator(Multiply.getMultiplyFnObject()), new MatrixBlock[]{left, right},new ScalarObject[]{}, new MatrixBlock());
			}
			case Ba_aC -> {
				res = LibMatrixMult.matrixMult(left,right, new MatrixBlock(), _numThreads);
			}
			case aB_Ca -> {
				res = LibMatrixMult.matrixMult(right,left, new MatrixBlock(), _numThreads);
			}
			case Ba_Ca -> {
				ReorgOperator transpose = new ReorgOperator(SwapIndex.getSwapIndexFnObject(), _numThreads);
				right = right.reorgOperations(transpose, new MatrixBlock(), 0, 0, 0);
				res = LibMatrixMult.matrixMult(left,right, new MatrixBlock(), _numThreads);
			}
			case aB_aC -> {
				ReorgOperator transpose = new ReorgOperator(SwapIndex.getSwapIndexFnObject(), _numThreads);
				left = left.reorgOperations(transpose, new MatrixBlock(), 0, 0, 0);
				res = LibMatrixMult.matrixMult(left,right, new MatrixBlock(), _numThreads);
			}
			case A_scalar, AB_scalar -> {
				res = MatrixBlock.naryOperations(new SimpleOperator(Multiply.getMultiplyFnObject()), new MatrixBlock[]{left},new ScalarObject[]{new DoubleObject(right.get(0,0))}, new MatrixBlock());
			}
			case BA_A -> {
				EnsureMatrixBlockRowVector(right);
				res = left.binaryOperations(new BinaryOperator(Multiply.getMultiplyFnObject()), right);
			}
			case Ba_a -> {
				EnsureMatrixBlockRowVector(right);
				res = left.binaryOperations(new BinaryOperator(Multiply.getMultiplyFnObject()), right);
				AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceCol.getReduceColFnObject(), _numThreads);
				res = (MatrixBlock) res.aggregateUnaryOperations(aggun, new MatrixBlock(), 0, null);
			}

			case AB_A -> {
				EnsureMatrixBlockColumnVector(right);
				res = left.binaryOperations(new BinaryOperator(Multiply.getMultiplyFnObject()), right);
			}
			case aB_a -> {
				EnsureMatrixBlockColumnVector(right);
				res = left.binaryOperations(new BinaryOperator(Multiply.getMultiplyFnObject()), right);
				AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceRow.getReduceRowFnObject(), _numThreads);
				res = (MatrixBlock) res.aggregateUnaryOperations(aggun, new MatrixBlock(), 0, null);
				EnsureMatrixBlockColumnVector(res);
			}

			case A_B -> {
				EnsureMatrixBlockColumnVector(left);
				EnsureMatrixBlockRowVector(right);
				res = left.binaryOperations(new BinaryOperator(Multiply.getMultiplyFnObject()), right);
			}
			case scalar_scalar -> {
				return new MatrixBlock(left.get(0,0)*right.get(0,0));
			}
			default -> {
				throw new IllegalArgumentException("Unexpected value: " + bin.operand.toString());
			}

		}
		return res;
	}

	private static MatrixBlock ComputeEOpNodeCodegen(EOpNode eOpNode, ArrayList<MatrixBlock> inputs){
		return rComputeEOpNodeCodegen(eOpNode, inputs);
//		throw new NotImplementedException();
	}
	private static CNodeData MatrixBlockToCNodeData(MatrixBlock mb, int id){
		return new CNodeData("ce"+id, id, mb.getNumRows(), mb.getNumColumns(), DataType.MATRIX);
	}
	private static MatrixBlock rComputeEOpNodeCodegen(EOpNode eOpNode, ArrayList<MatrixBlock> inputs) {
		if (eOpNode instanceof EOpNodeData eOpNodeData){
			return inputs.get(eOpNodeData.matrixIdx);
//			return new CNodeData("ce"+eOpNodeData.matrixIdx, eOpNodeData.matrixIdx, inputs.get(eOpNodeData.matrixIdx).getNumRows(), inputs.get(eOpNodeData.matrixIdx).getNumColumns(), DataType.MATRIX);
		}

		EOpNodeBinary bin = (EOpNodeBinary) eOpNode;
//		CNodeData dataLeft = null;
//		if (bin.left instanceof EOpNodeData eOpNodeData) dataLeft = new CNodeData("ce"+eOpNodeData.matrixIdx, eOpNodeData.matrixIdx, inputs.get(eOpNodeData.matrixIdx).getNumRows(), inputs.get(eOpNodeData.matrixIdx).getNumColumns(), DataType.MATRIX);
//		CNodeData dataRight = null;
//		if (bin.right instanceof EOpNodeData eOpNodeData) dataRight = new CNodeData("ce"+eOpNodeData.matrixIdx, eOpNodeData.matrixIdx, inputs.get(eOpNodeData.matrixIdx).getNumRows(), inputs.get(eOpNodeData.matrixIdx).getNumColumns(), DataType.MATRIX);

		if(bin.operand == EBinaryOperand.AB_AB){
			if (bin.right instanceof EOpNodeBinary rBinary && rBinary.operand  == EBinaryOperand.AB_AB){
				MatrixBlock left = rComputeEOpNodeCodegen(bin.left, inputs);

				MatrixBlock right1 = rComputeEOpNodeCodegen(((EOpNodeBinary) bin.right).left, inputs);
				MatrixBlock right2 = rComputeEOpNodeCodegen(((EOpNodeBinary) bin.right).right, inputs);

				CNodeData d0 = MatrixBlockToCNodeData(left, 0);
				CNodeData d1 = MatrixBlockToCNodeData(right1, 1);
				CNodeData d2 = MatrixBlockToCNodeData(right2, 2);
//				CNodeNary nary = new CNodeNary(cnodeIn, CNodeNary.NaryType.)
				CNodeBinary rightBinary = new CNodeBinary(d1, d2, CNodeBinary.BinType.VECT_MULT);
				CNodeBinary cNodeBinary = new CNodeBinary(d0, rightBinary, CNodeBinary.BinType.VECT_MULT);
				ArrayList<CNode> cnodeIn = new ArrayList<>();
				cnodeIn.add(d0);
				cnodeIn.add(d1);
				cnodeIn.add(d2);

				CNodeRow cnode = new CNodeRow(cnodeIn, cNodeBinary);

				cnode.setRowType(SpoofRowwise.RowType.NO_AGG);
				cnode.renameInputs();


				String src = cnode.codegen(false, SpoofCompiler.GeneratorAPI.JAVA);
				if( LOG.isTraceEnabled()) LOG.trace(CodegenUtils.printWithLineNumber(src));
				Class cla = CodegenUtils.compileClass("codegen." + cnode.getClassname(), src);

				SpoofOperator op = CodegenUtils.createInstance(cla);
				MatrixBlock mb = new MatrixBlock();

				ArrayList<ScalarObject> scalars = new ArrayList<>();
				ArrayList<MatrixBlock> mbs = new ArrayList<>(3);
				mbs.add(left);
				mbs.add(right1);
				mbs.add(right2);
				MatrixBlock	out = op.execute(mbs, scalars, mb, 6);

				return out;
			}
		}

		throw new NotImplementedException();
	}


	private void releaseMatrixInputs(ExecutionContext ec){
		for (CPOperand input : _in)
			if(input.getDataType()==DataType.MATRIX)
				ec.releaseMatrixInput(input.getName()); //todo release other
	}

	private static void EnsureMatrixBlockColumnVector(MatrixBlock mb){
		if(mb.getNumColumns() > 1){
			mb.setNumRows(mb.getNumColumns());
			mb.setNumColumns(1);
			mb.getDenseBlock().resetNoFill(mb.getNumRows(),1);
		}
	}
	private static void EnsureMatrixBlockRowVector(MatrixBlock mb){
		if(mb.getNumRows() > 1){
			mb.setNumColumns(mb.getNumRows());
			mb.setNumRows(1);
			mb.getDenseBlock().resetNoFill(1,mb.getNumColumns());
		}
	}

	private static void indent(StringBuilder sb, int level) {
		for (int i = 0; i < level; i++) {
			sb.append("  ");
		}
	}

	private MatrixBlock computeCellSummation(ArrayList<MatrixBlock> inputs, List<String> inputsChars, String resultString,
														   HashMap<Character, Integer> charToDimensionSizeInt, List<Character> summingChars, int outRows, int outCols){
		ArrayList<CNode> cnodeIn = new ArrayList<>();
		cnodeIn.add(new CNodeData(new LiteralOp(3), 0, 0, DataType.SCALAR));
		CNodeCell cnode = new CNodeCell(cnodeIn, null);
		StringBuilder sb = new StringBuilder();

		int indent = 2;
		indent(sb, indent);

		boolean needsSumming = summingChars.stream().anyMatch(x -> x != null);

		String itVar0 = cnode.createVarname();
		String outVar = itVar0;
		if (needsSumming) {
			sb.append("double ");
			sb.append(outVar);
			sb.append("=0;\n");
		}

		HashSet<Character> summedCharacters = new HashSet<>();
		Iterator<Character> hsIt = summingChars.iterator();
		while (hsIt.hasNext()) {
			indent(sb, indent);
			indent++;
			Character c = hsIt.next();
			String itVar = itVar0 + c;
			sb.append("for(int ");
			sb.append(itVar);
			sb.append("=0;");
			sb.append(itVar);
			sb.append("<");
			sb.append(charToDimensionSizeInt.get(c));
			sb.append(";");
			sb.append(itVar);
			sb.append("++){\n");
		}
		indent(sb, indent);
		if (needsSumming) {
			sb.append(outVar);
			sb.append("+=");
		}

		for (int i = 0; i < inputsChars.size(); i++) {
			if (inputsChars.get(i).length() == 0){
				sb.append("getValue(b[");
				sb.append(i);
				sb.append("],b[");
				sb.append(i);
				sb.append("].clen, 0,");
			}

			else if (summingChars.contains(inputsChars.get(i).charAt(0))) {
				sb.append("getValue(b[");
				sb.append(i);
				sb.append("],b[");
				sb.append(i);
				sb.append("].clen,");
				sb.append(itVar0);
				sb.append(inputsChars.get(i).charAt(0));
				sb.append(",");
			} else if (resultString.length() >= 1  && inputsChars.get(i).charAt(0) == resultString.charAt(0)) {
				sb.append("getValue(b[");
				sb.append(i);
				sb.append("],b[");
				sb.append(i);
				sb.append("].clen, rix,");
			} else if (resultString.length() == 2 && inputsChars.get(i).charAt(0) == resultString.charAt(1)) {
				sb.append("getValue(b[");
				sb.append(i);
				sb.append("],b[");
				sb.append(i);
				sb.append("].clen, cix,");
			} else {
				sb.append("getValue(b[");
				sb.append(i);
				sb.append("],b[");
				sb.append(i);
				sb.append("].clen, 0,");
			}

			if (inputsChars.get(i).length() != 2){
				sb.append("0)");
			}
			else if (summingChars.contains(inputsChars.get(i).charAt(1))) {
				sb.append(itVar0);
				sb.append(inputsChars.get(i).charAt(1));
				sb.append(")");
			} else if (resultString.length() >= 1 &&inputsChars.get(i).charAt(1) == resultString.charAt(0)) {
				sb.append("rix)");
			} else if (resultString.length() == 2  && inputsChars.get(i).charAt(1) == resultString.charAt(1)) {
				sb.append("cix)");
			} else {
				sb.append("0)");
			}

			if (i < inputsChars.size() - 1) {
				sb.append(" * ");
			}

		}
		if (needsSumming) {
			sb.append(";\n");
		}
		indent--;
		for (int si = 0; si < summingChars.size(); si++) {
			indent(sb, indent);
			indent--;
			sb.append("}\n");
		}
		String src = CNodeCell.JAVA_TEMPLATE;//
		src = src.replace("%TMP%", cnode.createVarname());
		src = src.replace("%TYPE%", "NO_AGG");
		src = src.replace("%SPARSE_SAFE%", "false");
		src = src.replace("%SEQ%", "true");
		src = src.replace("%AGG_OP_NAME%", "null");
		if (needsSumming) {
			src = src.replace("%BODY_dense%", sb.toString());
			src = src.replace("%OUT%", outVar);
		} else {
			src = src.replace("%BODY_dense%", "");
			src = src.replace("%OUT%", sb.toString());
		}

		if( LOG.isTraceEnabled()) LOG.trace(src);
		Class cla = CodegenUtils.compileClass("codegen." + cnode.getClassname(), src);
		SpoofOperator op = CodegenUtils.createInstance(cla);
		MatrixBlock resBlock = new MatrixBlock();
		resBlock.reset(outRows, outCols);
		inputs.add(0, resBlock);
		MatrixBlock out = op.execute(inputs, new ArrayList<>(), new MatrixBlock(), _numThreads);

		return out;
	}

	public CPOperand[] getInputs() {
		return _in;
	}
}
