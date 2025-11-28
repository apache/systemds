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

import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.lang3.tuple.Triple;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.hops.LiteralOp;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.hops.codegen.cplan.CNode;
import org.apache.sysds.hops.codegen.cplan.CNodeCell;
import org.apache.sysds.hops.codegen.cplan.CNodeData;
import org.apache.sysds.runtime.codegen.*;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.einsum.*;
import org.apache.sysds.runtime.einsum.EOpNodeBinary.EBinaryOperand;
import org.apache.sysds.runtime.functionobjects.*;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.matrix.operators.ReorgOperator;
import org.apache.sysds.utils.Explain;

import java.util.*;

import static org.apache.sysds.api.DMLScript.EXPLAIN;
import static org.apache.sysds.hops.rewrite.RewriteMatrixMultChainOptimization.mmChainDP;

public class EinsumCPInstruction extends BuiltinNaryCPInstruction {
    public static final boolean FORCE_CELL_TPL = false;

	public static final boolean FUSE_OUTER_MULTIPLY = true;
	public static final boolean FUSE_OUTER_MULTIPLY_EXCEEDS_L2_CACHE_CHECK = true;


	public static final boolean PRINT_TRACE = false;

	protected static final Log LOG = LogFactory.getLog(EinsumCPInstruction.class.getName());
	public String eqStr;
	private final int _numThreads;
	private final CPOperand[] _in;

	public EinsumCPInstruction(Operator op, String opcode, String istr, CPOperand out, CPOperand... inputs)
	{
		super(op, opcode, istr, out, inputs);
        _numThreads = OptimizerUtils.getConstrainedNumThreads(-1)/2;
		_in = inputs;
		this.eqStr = inputs[0].getName();
		if (PRINT_TRACE) {
//			System.out.println("fusing outer mult:"+FUSE_OUTER_MULTIPLY);
			Logger.getLogger(EinsumCPInstruction.class).setLevel(Level.TRACE);
		}
		else
        	Logger.getLogger(EinsumCPInstruction.class).setLevel(Level.WARN);
	}

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
                if(mb.getNumRows() == 1){
                    ensureMatrixBlockColumnVector(mb);
                }
				inputs.add(mb);
			}
		}

		EinsumContext einc = EinsumContext.getEinsumContext(eqStr, inputs);

		String resultString = einc.outChar2 != null ? String.valueOf(einc.outChar1) + einc.outChar2 : einc.outChar1 != null ? String.valueOf(einc.outChar1) : "";

		if( LOG.isTraceEnabled() ) LOG.trace("output: "+resultString +" "+einc.outRows+"x"+einc.outCols);

		ArrayList<String> inputsChars = einc.newEquationStringInputsSplit;

		if(LOG.isTraceEnabled()) LOG.trace(String.join(",",einc.newEquationStringInputsSplit));
		ArrayList<EOpNode> eOpNodes = new ArrayList<>(inputsChars.size());
		ArrayList<EOpNode> eOpNodesScalars = new ArrayList<>(inputsChars.size()); // computed separately and not included into plan until it is already created

		//make all vetors col vectors
		for(int i = 0; i < inputs.size(); i++){
			if(inputsChars.get(i).length() == 1) ensureMatrixBlockColumnVector(inputs.get(i));
		}

		addSumDimensionsDiagonalsAndScalars(einc, inputsChars, eOpNodes, eOpNodesScalars, einc.charToDimensionSize);

		HashMap<Character, Integer> characterToOccurences = einc.characterAppearanceCount;

		for (int i = 0; i < inputsChars.size(); i++) {
			if (inputsChars.get(i) == null) continue;
			Character c1 = inputsChars.get(i).isEmpty() ? null : inputsChars.get(i).charAt(0);
			Character c2 = inputsChars.get(i).length() > 1 ? inputsChars.get(i).charAt(1) : null;
			Integer dim1 = c1 == null ? null : einc.charToDimensionSize.get(c1);
			Integer dim2 = c1 == null ? null : einc.charToDimensionSize.get(c2);
			EOpNodeData n = new EOpNodeData(c1,c2,dim1,dim2, i);
			eOpNodes.add(n);
		}

		ArrayList<EOpNode> ret = new ArrayList<>();
		addVectorMultiplies(eOpNodes, eOpNodesScalars,characterToOccurences, einc.outChar1, einc.outChar2, ret);
		eOpNodes = ret;

		List<EOpNode> plan;
		ArrayList<MatrixBlock> remainingMatrices;

        if(!FORCE_CELL_TPL) {
			if(true){ // new way: search for fusions and matrix-multiplications chain in a loop
				plan = generatePlanFusionAndMM(eOpNodes, eOpNodesScalars, einc.charToDimensionSize, characterToOccurences, einc.outChar1, einc.outChar2);
			}else { // old way: try to do fusion first and then rest in binary fashion cost based
				List<EOpNodeFuse> fuseOps;
				do {
					ret = new ArrayList<>();
					fuseOps = EOpNodeFuse.findFuseOps(eOpNodes, einc.outChar1, einc.outChar2, einc.charToDimensionSize, characterToOccurences, ret);

					if(!fuseOps.isEmpty()) {
						for (EOpNodeFuse fuseOp : fuseOps) {
							if (fuseOp.c1 == null) {
								eOpNodesScalars.add(fuseOp);
								continue;
							}
							ret.add(fuseOp);
//							if (fuseOp.c2 != null) {
//								characterToOccurences.put(fuseOp.c2, characterToOccurences.get(fuseOp.c2)+1);
//							}
//							characterToOccurences.put(fuseOp.c1, characterToOccurences.get(fuseOp.c1)+1);
						}
						eOpNodes = ret;
					}
				} while(eOpNodes.size() > 1 && !fuseOps.isEmpty());

				Pair<Integer, List<EOpNode>> costAndPlan = generateBinaryPlanCostBased(0, eOpNodes, einc.charToDimensionSize, characterToOccurences,
					einc.outChar1, einc.outChar2);
				plan = costAndPlan.getRight();
			}
			if(!eOpNodesScalars.isEmpty()){
				EOpNode l = eOpNodesScalars.get(0);
				for(int i = 1; i < eOpNodesScalars.size(); i++){
					l = new EOpNodeBinary(l, eOpNodesScalars.get(i), EBinaryOperand.scalar_scalar);
				}

				if(plan.isEmpty()) plan.add(l);
				else {
					int minCost = Integer.MAX_VALUE;
					EOpNode addToNode = null;
					int minIdx = -1;
					for(int i = 0; i < plan.size(); i++) {
						EOpNode n = plan.get(i);
						Pair<Integer, EOpNode> costAndNode = addScalarToPlanFindMinCost(n, einc.charToDimensionSize);
						if(costAndNode.getLeft() < minCost) {
							minCost = costAndNode.getLeft();
							addToNode = costAndNode.getRight();
							minIdx = i;
						}
					}
					plan.set(minIdx, mergeEOpNodeWithScalar(addToNode, l));
				}

			}

			if(plan.size() == 2 && plan.get(0).c2 == null && plan.get(1).c2 == null){
				if (plan.get(0).c1 == einc.outChar1 && plan.get(1).c1 == einc.outChar2)
					plan.set(0, new EOpNodeBinary(plan.get(0), plan.get(1), EBinaryOperand.A_B));
				if (plan.get(0).c1 == einc.outChar2 && plan.get(1).c1 == einc.outChar1)
					plan.set(0, new EOpNodeBinary(plan.get(1), plan.get(0), EBinaryOperand.A_B));
				plan.remove(1);
			}

			if(plan.size() == 1)
				plan.set(0,plan.get(0).reorderChildrenAndOptimize(null, einc.outChar1, einc.outChar2));

			if (EXPLAIN != Explain.ExplainType.NONE ) {
				System.out.println("Einsum plan:");
				for(int i = 0; i < plan.size(); i++) {
					System.out.println((i + 1) + ".");
					System.out.println("- " + String.join("\n- ", plan.get(i).recursivePrintString()));
				}
			}

			remainingMatrices = executePlan(plan, inputs);
        }else{
			plan = eOpNodes;
			remainingMatrices = inputs;
		}



		if(!FORCE_CELL_TPL && remainingMatrices.size() == 1){
			EOpNode resNode = plan.get(0);
			if (einc.outChar1 != null && einc.outChar2 != null){
				if(resNode.c1 == einc.outChar1 && resNode.c2 == einc.outChar2){
					ec.setMatrixOutput(output.getName(), remainingMatrices.get(0));
				}
				else if(resNode.c1 == einc.outChar2 && resNode.c2 == einc.outChar1){
                    if( LOG.isTraceEnabled()) LOG.trace("Transposing the final result");

					ReorgOperator transpose = new ReorgOperator(SwapIndex.getSwapIndexFnObject(), _numThreads);
					MatrixBlock resM = remainingMatrices.get(0).reorgOperations(transpose, new MatrixBlock(),0,0,0);
					ec.setMatrixOutput(output.getName(), resM);
				}else{
					if(LOG.isTraceEnabled()) LOG.trace("Einsum error, expected: "+resultString + ", got: "+resNode.c1+resNode.c2);
					throw new RuntimeException("Einsum plan produced different result, expected: "+resultString + ", got: "+resNode.c1+resNode.c2);
				}
			}else if (einc.outChar1 != null){
				if(resNode.c1 == einc.outChar1  && resNode.c2 == null){
					ensureMatrixBlockColumnVector(remainingMatrices.get(0));
					ec.setMatrixOutput(output.getName(), remainingMatrices.get(0));
				}else{
					if(LOG.isTraceEnabled()) LOG.trace("Einsum expected: "+resultString + ", got: "+resNode.c1+resNode.c2);
					throw new RuntimeException("Einsum plan produced different result");
				}
			}else{
				if(resNode.c1 == null && resNode.c2 == null){
					ec.setScalarOutput(output.getName(), new DoubleObject(remainingMatrices.get(0).get(0, 0)));;
				}
			}
		}else{
			// use cell template with loops for remaining
			ArrayList<MatrixBlock> mbs = remainingMatrices;
			ArrayList<String> chars = new ArrayList<>();

			for (int i = 0; i < plan.size(); i++) {
				String s;
				if(plan.get(i).c1 == null) s = "";
				else if(plan.get(i).c2 == null) s = plan.get(i).c1.toString();
				else s = plan.get(i).c1.toString() + plan.get(i).c2;
				chars.add(s);
			}

			ArrayList<Character> summingChars = new ArrayList<>();
			for (Character c : characterToOccurences.keySet()) {
				if (c != einc.outChar1 && c != einc.outChar2) summingChars.add(c);
			}
			if(LOG.isTraceEnabled()) LOG.trace("finishing with cell tpl: "+String.join(",", chars));

			MatrixBlock res = computeCellSummation(mbs, chars, resultString, einc.charToDimensionSize, summingChars, einc.outRows, einc.outCols);

			if (einc.outChar2 == null)
				ensureMatrixBlockColumnVector(res);

			if (einc.outRows == 1 && einc.outCols == 1)
				ec.setScalarOutput(output.getName(), new DoubleObject(res.get(0, 0)));
			else ec.setMatrixOutput(output.getName(), res);
		}
		if(LOG.isTraceEnabled()) LOG.trace("EinsumCPInstruction Finished");

		releaseMatrixInputs(ec);

	}

	private EOpNode mergeEOpNodeWithScalar(EOpNode addToNode, EOpNode scalar) {
		if(addToNode instanceof EOpNodeFuse fuse) {
			switch (fuse.einsumRewriteType) {
				case AB_BA_B_A__A, AB_BA_B_A_AZ__Z -> {
					fuse.addScalarAsIntermediate(scalar);
					return fuse;
				}
			};
			return new EOpNodeBinary(addToNode,scalar,EBinaryOperand.AB_scalar);
		}
		if(addToNode.c1 == null)
			return new EOpNodeBinary(addToNode,scalar,EBinaryOperand.scalar_scalar);
		if(addToNode.c2 == null)
			return new EOpNodeBinary(addToNode,scalar,EBinaryOperand.A_scalar);
		return new EOpNodeBinary(addToNode,scalar,EBinaryOperand.AB_scalar);
	}

	private static Pair<Integer, EOpNode> addScalarToPlanFindMinCost(EOpNode plan, HashMap<Character, Integer> charToSizeMap) {
		int thisSize = 0;
		if(plan.c1 != null) thisSize += charToSizeMap.get(plan.c1);
		if(plan.c2 != null) thisSize += charToSizeMap.get(plan.c2);
		int cost = thisSize;

		if (plan instanceof EOpNodeData || plan instanceof EOpNodeUnary) return Pair.of(thisSize, plan);

		List<EOpNode> inputs = List.of();

		if (plan instanceof EOpNodeBinary bin) inputs = List.of(bin.left, bin.right);
		else if(plan instanceof EOpNodeFuse fuse){
			cost = switch (fuse.einsumRewriteType) {
				case AB_BA_B_A__ -> 1; // thisSize
				case AB_BA_B_A__AB -> thisSize;
				case AB_BA_A__B -> thisSize;
				case AB_BA_B_A__A -> 2; // intermediate is scalar, 2 because if there is some real scalar
				case AB_BA_B_A_AZ__Z -> 2; // intermediate is scalar
				case AB_BA_A_AZ__BZ -> thisSize;
				case AB_BA_A_AZ__ZB -> thisSize;
			};
			inputs = fuse.getAllOps();
		}

		for(EOpNode inp : inputs){
			Pair<Integer, EOpNode> min = addScalarToPlanFindMinCost(inp, charToSizeMap);
			if(min.getLeft() < cost){
				cost = min.getLeft();
				plan = min.getRight();
			}
		}
		return Pair.of(cost, plan);
	}

	private static void addVectorMultiplies(ArrayList<EOpNode> eOpNodes, ArrayList<EOpNode> eOpNodesScalars,HashMap<Character, Integer> charToOccurences, Character outChar1, Character outChar2,ArrayList<EOpNode> ret) {
		HashMap<Character, ArrayList<EOpNode>> vectorCharacterToIndices = new HashMap<>();
		for (int i = 0; i < eOpNodes.size(); i++) {
			if (eOpNodes.get(i).c2 == null) {
				if (vectorCharacterToIndices.containsKey(eOpNodes.get(i).c1))
					vectorCharacterToIndices.get(eOpNodes.get(i).c1).add(eOpNodes.get(i));
				else
					vectorCharacterToIndices.put(eOpNodes.get(i).c1, new ArrayList<>(Collections.singletonList(eOpNodes.get(i))));
			}
		}
		HashSet<EOpNode> usedNodes = new HashSet<>();
		for(Character c : vectorCharacterToIndices.keySet()){
			ArrayList<EOpNode> nodes = vectorCharacterToIndices.get(c);

			if(nodes.size()==1) continue;
			EOpNode left = nodes.get(0);
			usedNodes.add(left);
			boolean canBeSummed = c != outChar1 && c != outChar2 && charToOccurences.get(c) == nodes.size();

			for(int i = 1; i < nodes.size(); i++){
				EOpNode right = nodes.get(i);

				if(canBeSummed && i == nodes.size()-1){
					left = new EOpNodeBinary(left,right, EBinaryOperand.a_a);
				}else {
					left = new EOpNodeBinary(left,right, EBinaryOperand.A_A);
				}
				usedNodes.add(right);
			}
			if(canBeSummed) {
				eOpNodesScalars.add(left);
				charToOccurences.put(c, 0);
			}
			else {
				ret.add(left);
				charToOccurences.put(c, charToOccurences.get(c) - nodes.size() + 1);
			}
		}
		for(EOpNode inp : eOpNodes){
			if(!usedNodes.contains(inp)) ret.add(inp);
		}
	}

	private void addSumDimensionsDiagonalsAndScalars(EinsumContext einc, ArrayList<String> inputStrings,
		ArrayList<EOpNode> eOpNodes, ArrayList<EOpNode> eOpNodesScalars,
		HashMap<Character, Integer> charToDimensionSize) {
		for(int i = 0; i< inputStrings.size(); i++){
			String s = inputStrings.get(i);
			if (s.length() == 0){
				eOpNodesScalars.add(new EOpNodeData(null, null, null, null,i));
				inputStrings.set(i, null);
				continue;
			}else if (s.length() == 1){
				char c1 = s.charAt(0);
				if((einc.outChar1 == null || c1 != einc.outChar1) && (einc.outChar2 == null || c1 != einc.outChar2) && einc.characterAppearanceCount.get(c1) == 1){
					EOpNode e0 = new EOpNodeData(c1, null, charToDimensionSize.get(c1), null, i);
					eOpNodesScalars.add(new EOpNodeUnary(null, null, null, null, e0, EOpNodeUnary.EUnaryOperand.SUM));
					inputStrings.set(i, null);
				}
				continue;
			}

			char c1 = s.charAt(0);
			char c2 = s.charAt(1);
			Character newC1 = null;
			EOpNodeUnary.EUnaryOperand op = null;

			if(c1 == c2){
				if((einc.outChar1 == null || c1 != einc.outChar1) && (einc.outChar2 == null || c1 != einc.outChar2) && einc.characterAppearanceCount.get(c1) == 2){
					op = EOpNodeUnary.EUnaryOperand.SUM;
				}else {
					einc.characterAppearanceCount.put(c1, einc.characterAppearanceCount.get(c1) - 1);
					op = EOpNodeUnary.EUnaryOperand.DIAG;
					newC1 = c1;
				}
			}else if((einc.outChar1 == null || c1 != einc.outChar1) && (einc.outChar2 == null || c1 != einc.outChar2) && einc.characterAppearanceCount.get(c1) == 1){
				if ((einc.outChar1 == null || c2 != einc.outChar1) && (einc.outChar2 == null || c2 != einc.outChar2) && einc.characterAppearanceCount.get(c2) == 1){
					op = EOpNodeUnary.EUnaryOperand.SUM;
				}else{
					newC1 = c2;
					op = EOpNodeUnary.EUnaryOperand.SUM_COLS;
				}
			}else if((einc.outChar1 == null || c2 != einc.outChar1) && (einc.outChar2 == null || c2 != einc.outChar2) && einc.characterAppearanceCount.get(c2) == 1){
				newC1 =  c1;
				op = EOpNodeUnary.EUnaryOperand.SUM_ROWS;
			}

			if(op == null) continue;
			EOpNodeData e0 = new EOpNodeData(c1, c2, charToDimensionSize.get(c1), charToDimensionSize.get(c2),  i);
			Integer dim1 = newC1 == null ? null : charToDimensionSize.get(newC1);
			EOpNodeUnary res = new EOpNodeUnary(newC1, null, dim1, null, e0, op);

			if(op == EOpNodeUnary.EUnaryOperand.SUM) eOpNodesScalars.add(res);
			else eOpNodes.add(res);

			inputStrings.set(i, null);
		}
	}

	private static List<EOpNode> generatePlanFusionAndMM(ArrayList<EOpNode> eOpNodes,
		ArrayList<EOpNode> eOpNodesScalars, HashMap<Character, Integer> charToSizeMap, HashMap<Character, Integer> charToOccurences, Character outChar1, Character outChar2) {
		ArrayList<EOpNode> ret;
		int lastNumOfOperands = -1;
		while(lastNumOfOperands != eOpNodes.size() && eOpNodes.size() > 1){
			lastNumOfOperands = eOpNodes.size();

			List<EOpNodeFuse> fuseOps;
			do {
				ret = new ArrayList<>();
				fuseOps = EOpNodeFuse.findFuseOps(eOpNodes, outChar1, outChar2, charToSizeMap, charToOccurences, ret);

				if(!fuseOps.isEmpty()) {
					for (EOpNodeFuse fuseOp : fuseOps) {
						if (fuseOp.c1 == null) {
							eOpNodesScalars.add(fuseOp);
							continue;
						}
						ret.add(fuseOp);
//						if (fuseOp.c2 != null) {
//							charToOccurences.put(fuseOp.c2, charToOccurences.get(fuseOp.c2)+1);
//						}
//						charToOccurences.put(fuseOp.c1, charToOccurences.get(fuseOp.c1)+1);
					}
					eOpNodes = ret;
				}
			} while(eOpNodes.size() > 1 && !fuseOps.isEmpty());

			ret = new ArrayList<>();
			addVectorMultiplies(eOpNodes, eOpNodesScalars,charToOccurences, outChar1, outChar2, ret);
			eOpNodes = ret;

			ret = new ArrayList<>();
			ArrayList<List<EOpNode>> matrixMultiplies = findMatrixMultiplicationChains(eOpNodes, outChar1, outChar2, charToOccurences,
				ret);

			for(List<EOpNode> list : matrixMultiplies) {
				EOpNodeBinary bin = optimizeMMChain(list, charToSizeMap);
				ret.add(bin);
			}
			eOpNodes = ret;
		}

		return eOpNodes;
	}

	private static EOpNodeBinary optimizeMMChain(List<EOpNode> mmChain, HashMap<Character, Integer> charToSizeMap) {
		ArrayList<Pair<Integer, Integer>> dimensions = new ArrayList<>();

		for(int i = 0; i < mmChain.size()-1; i++){
			EOpNode n1 = mmChain.get(i);
			EOpNode n2 = mmChain.get(i+1);
			if(n1.c2 == n2.c1 || n1.c2 == n2.c2) dimensions.add(Pair.of(charToSizeMap.get(n1.c1), charToSizeMap.get(n1.c2)));
			else dimensions.add(Pair.of(charToSizeMap.get(n1.c2), charToSizeMap.get(n1.c1))); // transpose this one
		}
		EOpNode prelast = mmChain.get(mmChain.size()-2);
		EOpNode last = mmChain.get(mmChain.size()-1);
		if (last.c1 == prelast.c2 || last.c1 == prelast.c1) dimensions.add(Pair.of(charToSizeMap.get(last.c1), charToSizeMap.get(last.c2)));
		else dimensions.add(Pair.of(charToSizeMap.get(last.c2), charToSizeMap.get(last.c1)));


		double[] dimsArray = new double[mmChain.size() + 1];
		getDimsArray( dimensions, dimsArray );

		int size = mmChain.size();
		int[][] splitMatrix = mmChainDP(dimsArray, mmChain.size());

		EOpNodeBinary res = (EOpNodeBinary) getBinaryFromSplit(splitMatrix,0,size-1, mmChain);
		return res;
	}

	private static EOpNode getBinaryFromSplit(int[][] splitMatrix, int i, int j, List<EOpNode> mmChain) {
		if (i==j) return mmChain.get(i);
		int split =  splitMatrix[i][j];

		EOpNode left = getBinaryFromSplit(splitMatrix,i,split,mmChain);
		EOpNode right = getBinaryFromSplit(splitMatrix,split+1,j,mmChain);
		return EOpNodeBinary.combineMatrixMultiply(left, right);
	}

	private static void getDimsArray( ArrayList<Pair<Integer, Integer>> chain, double[] dimsArray )
	{
		for( int i = 0; i < chain.size(); i++ ) {
			if (i == 0) {
				dimsArray[i] = chain.get(i).getLeft();
				if (dimsArray[i] <= 0) {
					throw new RuntimeException(
						"EinsumCPInstruction::optimizeMMChain() : Invalid Matrix Dimension: "+ dimsArray[i]);
				}
			}
			else if (chain.get(i - 1).getRight() != chain.get(i).getLeft()) {
				throw new RuntimeException(
					"EinsumCPInstruction::optimizeMMChain() : Matrix Dimension Mismatch: " +
					chain.get(i - 1).getRight()+" != "+chain.get(i).getLeft());
			}

			dimsArray[i + 1] = chain.get(i).getRight();
			if( dimsArray[i + 1] <= 0 ) {
				throw new RuntimeException(
					"EinsumCPInstruction::optimizeMMChain() : Invalid Matrix Dimension: " + dimsArray[i + 1]);
			}
		}
	}
	private static ArrayList<List<EOpNode>> findMatrixMultiplicationChains(ArrayList<EOpNode> inpOperands, Character outChar1, Character outChar2, HashMap<Character, Integer> charToOccurences,
		ArrayList<EOpNode> ret) {
		HashSet<Character> charactersThatCanBeContracted = new HashSet<>();
		HashMap<Character, ArrayList<EOpNode>> characterToNodes = new HashMap<>();
		ArrayList<EOpNode> operandsTodo =  new ArrayList<>();
		for(EOpNode op : inpOperands) {
			if(op.c2 == null || op.c1 == null) continue;

			if (characterToNodes.containsKey(op.c1))  characterToNodes.get(op.c1).add(op);
			else characterToNodes.put(op.c1, new ArrayList<>(Collections.singletonList(op)));
			if (characterToNodes.containsKey(op.c2)) characterToNodes.get(op.c2).add(op);
			else characterToNodes.put(op.c2, new ArrayList<>(Collections.singletonList(op)));

			boolean todo = false;
			if (charToOccurences.get(op.c1) == 2 && op.c1 != outChar1 && op.c1 != outChar2) {
				charactersThatCanBeContracted.add(op.c1);
				todo = true;
			}
			if (charToOccurences.get(op.c2) == 2 && op.c2 != outChar1 && op.c2 != outChar2) {
				charactersThatCanBeContracted.add(op.c2);
				todo = true;
			}
			if (todo)  operandsTodo.add(op);
		}
		ArrayList<List<EOpNode>> res = new ArrayList<>();

		HashSet<EOpNode> doneNodes = new HashSet<>();

		for(int i = 0; i < operandsTodo.size(); i++){
			EOpNode iterateNode = operandsTodo.get(i);

			if (doneNodes.contains(iterateNode)) continue;// was added previously somewhere
			doneNodes.add(iterateNode);

			LinkedList<EOpNode> multiplies = new LinkedList<>();
			multiplies.add(iterateNode);

			EOpNode nextNode = iterateNode;
			Character nextC = iterateNode.c2;
			// add to right using c2
			while(charactersThatCanBeContracted.contains(nextC)) {
				EOpNode one = characterToNodes.get(nextC).get(0);
				EOpNode two = characterToNodes.get(nextC).get(1);
				if (nextNode == one){
					multiplies.addLast(two);
					nextNode = two;
				}else{
					multiplies.addLast(one);
					nextNode = one;
				}
				if(nextNode.c1 == nextC) nextC = nextNode.c2;
				else nextC = nextNode.c1;
				doneNodes.add(nextNode);
			}

			// add to left using c1
			nextNode = iterateNode;
			nextC = iterateNode.c1;
			while(charactersThatCanBeContracted.contains(nextC)) {
				EOpNode one = characterToNodes.get(nextC).get(0);
				EOpNode two = characterToNodes.get(nextC).get(1);
				if (nextNode == one){
					multiplies.addFirst(two);
					nextNode = two;
				}else{
					multiplies.addFirst(one);
					nextNode = one;
				}
				if(nextNode.c1 == nextC) nextC = nextNode.c2;
				else nextC = nextNode.c1;
				doneNodes.add(nextNode);
			}

			res.add(multiplies);
		}

		for(EOpNode op : inpOperands) {
			if (doneNodes.contains(op)) continue;
			ret.add(op);
		}

		return res;
	}

	// old way
	private Pair<Integer, List<EOpNode>> generateBinaryPlanCostBased(int cost, ArrayList<EOpNode> operands, HashMap<Character, Integer> charToSizeMap, HashMap<Character, Integer> charToOccurences, Character outChar1, Character outChar2) {
		Integer minCost = cost;
		List<EOpNode> minNodes = operands;

		if (operands.size() == 2){
			boolean swap = (operands.get(0).c2 == null && operands.get(1).c2 != null) || operands.get(0).c1 == null;
			EOpNode n1 = operands.get(!swap ? 0 : 1);
			EOpNode n2 = operands.get(!swap ? 1 : 0);
			Triple<Integer, EBinaryOperand, Pair<Character, Character>> t = EOpNodeBinary.TryCombineAndCost(n1, n2, charToSizeMap, charToOccurences, outChar1, outChar2);
			if (t != null) {
				EOpNodeBinary newNode = new EOpNodeBinary(n1, n2, t.getMiddle());
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


				Triple<Integer, EBinaryOperand, Pair<Character, Character>> t = EOpNodeBinary.TryCombineAndCost(n1, n2, charToSizeMap, charToOccurences, outChar1, outChar2);
				if (t != null){
					EOpNodeBinary newNode = new EOpNodeBinary(n1, n2, t.getMiddle());
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

					Pair<Integer, List<EOpNode>> furtherPlan = generateBinaryPlanCostBased(thisCost, newOperands,charToSizeMap, charToOccurences, outChar1, outChar2);
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

	private ArrayList<MatrixBlock> executePlan(List<EOpNode> plan, ArrayList<MatrixBlock> inputs) {
		ArrayList<MatrixBlock> res = new ArrayList<>(plan.size());
		for(EOpNode p : plan){
            res.add(p.computeEOpNode(inputs, _numThreads, LOG));
		}
		return res;
	}

	private void releaseMatrixInputs(ExecutionContext ec){
		for (CPOperand input : _in)
			if(input.getDataType()==DataType.MATRIX)
				ec.releaseMatrixInput(input.getName()); //todo release other
	}

	public static void ensureMatrixBlockColumnVector(MatrixBlock mb){
		if(mb.getNumColumns() > 1){
			mb.setNumRows(mb.getNumColumns());
			mb.setNumColumns(1);
			mb.getDenseBlock().resetNoFill(mb.getNumRows(),1);
		}
	}
    public static void ensureMatrixBlockRowVector(MatrixBlock mb){
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
		ArrayList<CNode> dummyIn = new ArrayList<>();
        dummyIn.add(new CNodeData(new LiteralOp(0), 0, 0, DataType.SCALAR));
		CNodeCell cnode = new CNodeCell(dummyIn, null);
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
			} else if (resultString.length() >= 1 && inputsChars.get(i).charAt(1) == resultString.charAt(0)) {
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
		String src = CNodeCell.JAVA_TEMPLATE;
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
		Class<?> cla = CodegenUtils.compileClass("codegen." + cnode.getClassname(), src);
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
