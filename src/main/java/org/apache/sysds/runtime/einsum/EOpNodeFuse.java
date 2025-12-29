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

package org.apache.sysds.runtime.einsum;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.logging.Log;
import org.apache.sysds.runtime.codegen.SpoofRowwise;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.functionobjects.SwapIndex;
import org.apache.sysds.runtime.instructions.cp.DoubleObject;
import org.apache.sysds.runtime.instructions.cp.EinsumCPInstruction;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.matrix.data.LibMatrixMult;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.ReorgOperator;
import org.jetbrains.annotations.NotNull;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.function.Function;

import static org.apache.sysds.runtime.instructions.cp.EinsumCPInstruction.ensureMatrixBlockColumnVector;
import static org.apache.sysds.runtime.instructions.cp.EinsumCPInstruction.ensureMatrixBlockRowVector;

public class EOpNodeFuse extends EOpNode {

	private EOpNode scalar = null;

	public enum EinsumRewriteType{
        // B -> row*vec, A -> row*scalar
        AB_BA_B_A__AB,
		AB_BA_A__B,
        AB_BA_B_A__A,
        AB_BA_B_A__,

        // scalar from row(AB).dot(B) multiplied by row(AZ)
        AB_BA_B_A_AZ__Z,

        // AZ: last step is outer matrix multiplication using vector Z
        AB_BA_A_AZ__BZ, AB_BA_A_AZ__ZB,
    }

    public EinsumRewriteType einsumRewriteType;
	public List<EOpNode> ABs;
	public List<EOpNode> BAs;
	public List<EOpNode> Bs;
	public List<EOpNode> As;
	public List<EOpNode> AZs;
	@Override
	public List<EOpNode> getChildren(){
		List<EOpNode> all = new ArrayList<>();
		all.addAll(ABs);
		all.addAll(BAs);
		all.addAll(Bs);
		all.addAll(As);
		all.addAll(AZs);
		if (scalar != null) all.add(scalar);
		return all;
	};
    private EOpNodeFuse(Character c1, Character c2, Integer dim1, Integer dim2, EinsumRewriteType einsumRewriteType, List<EOpNode> ABs, List<EOpNode> BAs, List<EOpNode> Bs, List<EOpNode> As, List<EOpNode> AZs) {
        super(c1,c2, dim1, dim2);
        this.einsumRewriteType = einsumRewriteType;
		this.ABs = ABs;
		this.BAs = BAs;
		this.Bs = Bs;
		this.As = As;
		this.AZs = AZs;
    }
	public EOpNodeFuse(EinsumRewriteType einsumRewriteType, List<EOpNode> ABs, List<EOpNode> BAs, List<EOpNode> Bs, List<EOpNode> As, List<EOpNode> AZs, List<Pair<List<EOpNode>, List<EOpNode>>> AXsAndXs) {
		super(null,null,null, null);
		switch(einsumRewriteType) {
			case AB_BA_B_A__A->{
				c1 = ABs.get(0).c1;
				dim1 = ABs.get(0).dim1;
			}case AB_BA_A__B -> {
				c1 = ABs.get(0).c2;
				dim1 = ABs.get(0).dim2;
			}case AB_BA_B_A__ -> {
			}case AB_BA_B_A__AB -> {
				c1 = ABs.get(0).c1;
				dim1 = ABs.get(0).dim1;
				c2 = ABs.get(0).c2;
				dim2 = ABs.get(0).dim2;
			}case AB_BA_B_A_AZ__Z -> {
				c1 = AZs.get(0).c1;
				dim1 = AZs.get(0).dim2;
			}case AB_BA_A_AZ__BZ ->{
				c1 = ABs.get(0).c2;
				dim1 = ABs.get(0).dim2;
				c2 = AZs.get(0).c2;
				dim2 = AZs.get(0).dim2;
			}case AB_BA_A_AZ__ZB ->{
				c2 = ABs.get(0).c2;
				dim2 = ABs.get(0).dim2;
				c1 = AZs.get(0).c2;
				dim1 = AZs.get(0).dim2;
			}
		}
		this.einsumRewriteType = einsumRewriteType;
		this.ABs = ABs;
		this.BAs = BAs;
		this.Bs = Bs;
		this.As = As;
		this.AZs = AZs;
	}

	@Override
	public String toString() {
		return this.getClass().getSimpleName()+" ("+einsumRewriteType.toString()+") "+this.getOutputString();
	}

	public void addScalarAsIntermediate(EOpNode scalar) {
		if(einsumRewriteType == EinsumRewriteType.AB_BA_B_A__A || einsumRewriteType == EinsumRewriteType.AB_BA_B_A_AZ__Z)
			this.scalar = scalar;
		else
			throw new RuntimeException("EOpNodeFuse.addScalarAsIntermediate: scalar is undefined for type "+einsumRewriteType.toString());
	}

    public static List<EOpNodeFuse> findFuseOps(ArrayList<EOpNode> operands, Character outChar1, Character outChar2,
		HashMap<Character, Integer> charToSize, HashMap<Character, Integer> charToOccurences, ArrayList<EOpNode> ret) {
		ArrayList<EOpNodeFuse> result = new ArrayList<>();
		HashSet<String> matricesChars = new HashSet<>();
		HashMap<Character, HashSet<String>> matricesCharsStartingWithChar = new HashMap<>();
		HashMap<String, ArrayList<EOpNode>> charsToMatrices = new HashMap<>();

		for(EOpNode operand1 : operands) {
			String k;

			if(operand1.c2 != null) {
				k = operand1.c1.toString() + operand1.c2;
				matricesChars.add(k);
				if(matricesCharsStartingWithChar.containsKey(operand1.c1)) {
					matricesCharsStartingWithChar.get(operand1.c1).add(k);
				}
				else {
					HashSet<String> set = new HashSet<>();
					set.add(k);
					matricesCharsStartingWithChar.put(operand1.c1, set);
				}
			}
			else {
				k = operand1.c1.toString();
			}

			if(charsToMatrices.containsKey(k)) {
				charsToMatrices.get(k).add(operand1);
			}
			else {
				ArrayList<EOpNode> matrices = new ArrayList<>();
				matrices.add(operand1);
				charsToMatrices.put(k, matrices);
			}
		}
		ArrayList<Pair<Integer, String>> matricesCharsSorted = new ArrayList<>(matricesChars.stream()
			.map(x -> Pair.of(charsToMatrices.get(x).get(0).dim1 * charsToMatrices.get(x).get(0).dim2, x)).toList());
		matricesCharsSorted.sort(Comparator.comparing(Pair::getLeft));
		ArrayList<EOpNode> AZs = new ArrayList<>();

		HashSet<String> usedMatricesChars = new HashSet<>();
		HashSet<EOpNode> usedOperands = new HashSet<>();

		for(String ABCandidate : matricesCharsSorted.stream().map(Pair::getRight).toList()) {
			if(usedMatricesChars.contains(ABCandidate)) continue;

			char a = ABCandidate.charAt(0);
			char b = ABCandidate.charAt(1);
			String AB = ABCandidate;
			String BA = "" + b + a;

			int BAsCount = (charsToMatrices.containsKey(BA) ? charsToMatrices.get(BA).size() : 0);
			int ABsCount = charsToMatrices.get(AB).size();

			if(BAsCount > ABsCount + 1) {
				BA = "" + a + b;
				AB = "" + b + a;
				char tmp = a;
				a = b;
				b = tmp;
				int tmp2 = ABsCount;
				ABsCount = BAsCount;
				BAsCount = tmp2;
			}
			String A = "" + a;
			String B = "" + b;
			int AsCount = (charsToMatrices.containsKey(A) && !usedMatricesChars.contains(A) ? charsToMatrices.get(A).size() : 0);
			int BsCount = (charsToMatrices.containsKey(B) && !usedMatricesChars.contains(B) ? charsToMatrices.get(B).size() : 0);

			if(AsCount == 0 && BsCount == 0 && (ABsCount + BAsCount) < 2) { // no elementwise multiplication possible
				continue;
			}

			int usedBsCount = BsCount + ABsCount + BAsCount;

			boolean doSumA = false;
			boolean doSumB = charToOccurences.get(b) == usedBsCount && (outChar1 == null || b != outChar1) && (outChar2 == null || b != outChar2);
			HashSet<String> AZCandidates = matricesCharsStartingWithChar.get(a);

			String AZ = null;
			Character z = null;
			boolean includeAZ = AZCandidates.size() == 2;

			if(includeAZ) {
				for(var AZCandidate : AZCandidates) {
					if(AB.equals(AZCandidate)) {continue;}
					AZs = charsToMatrices.get(AZCandidate);
					z = AZCandidate.charAt(1);
					String Z = "" + z;
					AZ = "" + a + z;
					int AZsCount= AZs.size();
					int ZsCount= charsToMatrices.containsKey(Z) ? charsToMatrices.get(Z).size() : 0;
					doSumA = AZsCount + ABsCount + BAsCount + AsCount == charToOccurences.get(a) && (outChar1 == null || a != outChar1) && (outChar2 == null || a != outChar2);
					boolean doSumZ = AZsCount + ZsCount  == charToOccurences.get(z) && (outChar1 == null || z != outChar1) && (outChar2 == null || z != outChar2);
					if(!doSumA){
						includeAZ = false;
					} else if(!doSumB && doSumZ){ // swap the order, to have only one fusion AB,...,AZ->Z
						b = z;
						z = AB.charAt(1);
						AB = "" + a + b;
						BA = "" + b + a;
						A = "" + a;
						B = "" + b;
						AZ = "" + a + z;
						AZs = charsToMatrices.get(AZ);
						doSumB = true;
					} else if(!doSumB && !doSumZ){ // outer between B and Z
						if(!EinsumCPInstruction.FUSE_OUTER_MULTIPLY
							|| (EinsumCPInstruction.FUSE_OUTER_MULTIPLY_EXCEEDS_L2_CACHE_CHECK && ((charToSize.get(a) * charToSize.get(b) *(ABsCount + BAsCount)) + (charToSize.get(a)*charToSize.get(z)*(AZsCount))) * 8 < 6 * 1024 * 1024)
							|| !LibMatrixMult.isSkinnyRightHandSide(charToSize.get(AB.charAt(0)), charToSize.get(AB.charAt(1)), charToSize.get(AB.charAt(0)), charToSize.get(AZCandidates.iterator().next().charAt(1)),false)) {
							includeAZ = false;
						}
					} else if(doSumB && doSumZ){
						// it will be two separate templates and then mutliply a vectors
					} else if (doSumB && !doSumZ) {
						// ->Z template OK
					}
					break;
				}
			}

			if(!includeAZ) {
				doSumA = charToOccurences.get(a) == AsCount + ABsCount + BAsCount && (outChar1 == null || a != outChar1) && (outChar2 == null || a != outChar2);
			}

			ArrayList<EOpNode> ABs = charsToMatrices.containsKey(AB) ? charsToMatrices.get(AB) : new ArrayList<>();
			ArrayList<EOpNode> BAs = charsToMatrices.containsKey(BA) ? charsToMatrices.get(BA) : new ArrayList<>();
			ArrayList<EOpNode> As = charsToMatrices.containsKey(A) && !usedMatricesChars.contains(A) ? charsToMatrices.get(A) : new ArrayList<>();
			ArrayList<EOpNode> Bs = charsToMatrices.containsKey(B) && !usedMatricesChars.contains(B) ? charsToMatrices.get(B) : new ArrayList<>();
			Character c1 = null, c2 = null;
			Integer dim1 = null, dim2 = null;
			EinsumRewriteType type = null;

			if(includeAZ) {
				if(doSumB) {
					type = EinsumRewriteType.AB_BA_B_A_AZ__Z;
					c1 = z;
				}
				else if((outChar1 != null && outChar2 != null) && outChar1 == z && outChar2 == b) {
					type = EinsumRewriteType.AB_BA_A_AZ__ZB;
					c1 = z; c2 = b;
				}
				else if((outChar1 != null && outChar2 != null) && outChar1 == b && outChar2 == z) {
					type = EinsumRewriteType.AB_BA_A_AZ__BZ;
					c1 = b; c2 = z;
				}
				else {
					type = EinsumRewriteType.AB_BA_A_AZ__ZB;
					c1 = z; c2 = b;
				}
			}
			else {
				AZs= new ArrayList<>();
				if(doSumA) {
					if(doSumB) {
						type = EinsumRewriteType.AB_BA_B_A__;
					}
					else {
						type = EinsumRewriteType.AB_BA_A__B;
						c1 = AB.charAt(1);
					}
				}
				else if(doSumB) {
					type = EinsumRewriteType.AB_BA_B_A__A;
					c1 = AB.charAt(0);
				}
				else {
					type = EinsumRewriteType.AB_BA_B_A__AB;
					c1 = AB.charAt(0); c2 = AB.charAt(1);
				}
			}

			if(c1 != null) {
				charToOccurences.put(c1, charToOccurences.get(c1) + 1);
				dim1 = charToSize.get(c1);
			}
			if(c2 != null) {
				charToOccurences.put(c2, charToOccurences.get(c2) + 1);
				dim2 = charToSize.get(c2);
			}
			boolean includeB = type != EinsumRewriteType.AB_BA_A__B && type != EinsumRewriteType.AB_BA_A_AZ__BZ && type != EinsumRewriteType.AB_BA_A_AZ__ZB;

			usedOperands.addAll(ABs);
			usedOperands.addAll(BAs);
			usedOperands.addAll(As);
			if (includeB) usedOperands.addAll(Bs);
			if (includeAZ) usedOperands.addAll(AZs);

			usedMatricesChars.add(AB);
			usedMatricesChars.add(BA);
			usedMatricesChars.add(A);
			if (includeB) usedMatricesChars.add(B);
			if (includeAZ) usedMatricesChars.add(AZ);

			var e = new EOpNodeFuse(c1, c2, dim1, dim2, type, ABs, BAs, includeB ? Bs : new ArrayList<>(), As, AZs);

			result.add(e);
		}

        for(EOpNode n : operands) {
            if(!usedOperands.contains(n)){
                ret.add(n);
            } else {
				charToOccurences.put(n.c1, charToOccurences.get(n.c1) - 1);
				if(charToOccurences.get(n.c2)!= null)
					charToOccurences.put(n.c2, charToOccurences.get(n.c2)-1);
			}
        }

        return result;
    }
	public static MatrixBlock compute(EinsumRewriteType rewriteType, List<MatrixBlock> ABsInput, List<MatrixBlock> mbBAs, List<MatrixBlock> mbBs, List<MatrixBlock> mbAs, List<MatrixBlock> mbAZs,
		Double scalar, int numThreads){
		boolean isResultAB =rewriteType  == EOpNodeFuse.EinsumRewriteType.AB_BA_B_A__AB;
		boolean isResultA = rewriteType == EOpNodeFuse.EinsumRewriteType.AB_BA_B_A__A;
		boolean isResultB = rewriteType == EOpNodeFuse.EinsumRewriteType.AB_BA_A__B;
		boolean isResult_ = rewriteType == EOpNodeFuse.EinsumRewriteType.AB_BA_B_A__;
		boolean isResultZ = rewriteType == EOpNodeFuse.EinsumRewriteType.AB_BA_B_A_AZ__Z;
		boolean isResultBZ =rewriteType  == EOpNodeFuse.EinsumRewriteType.AB_BA_A_AZ__BZ;
		boolean isResultZB =rewriteType  == EOpNodeFuse.EinsumRewriteType.AB_BA_A_AZ__ZB;

		ArrayList<MatrixBlock> mbABs = new ArrayList<>(ABsInput);
		int bSize = mbABs.get(0).getNumColumns();
		int aSize = mbABs.get(0).getNumRows();
		if (!mbBAs.isEmpty()) {
			ReorgOperator transpose = new ReorgOperator(SwapIndex.getSwapIndexFnObject(), numThreads);
			for(MatrixBlock mb : mbBAs) //BA->AB
				mbABs.add(mb.reorgOperations(transpose, null, 0, 0, 0));
		}

		if(mbAs.size() > 1) mbAs = multiplyVectorsIntoOne(mbAs, aSize);
		if(mbBs.size() > 1) mbBs = multiplyVectorsIntoOne(mbBs, bSize);

		int constDim2 = -1;
		int zSize = 0;
		int azCount = 0;
		switch(rewriteType){
			case AB_BA_B_A_AZ__Z, AB_BA_A_AZ__BZ, AB_BA_A_AZ__ZB ->  {
				constDim2 = mbAZs.get(0).getNumColumns();
				zSize = mbAZs.get(0).getNumColumns();
				azCount = mbAZs.size();
			}
		}

		SpoofRowwise.RowType rowType = switch(rewriteType){
			case AB_BA_B_A__AB -> SpoofRowwise.RowType.NO_AGG;
			case AB_BA_A__B -> SpoofRowwise.RowType.COL_AGG_T;
			case AB_BA_B_A__A -> SpoofRowwise.RowType.ROW_AGG;
			case AB_BA_B_A__ -> SpoofRowwise.RowType.FULL_AGG;
			case AB_BA_B_A_AZ__Z -> SpoofRowwise.RowType.COL_AGG_CONST;
			case AB_BA_A_AZ__BZ -> SpoofRowwise.RowType.COL_AGG_B1_T;
			case AB_BA_A_AZ__ZB -> SpoofRowwise.RowType.COL_AGG_B1;
		};
		EinsumSpoofRowwise r = new EinsumSpoofRowwise(rewriteType, rowType, constDim2,
			mbABs.size()-1, !mbBs.isEmpty() && (!isResultBZ && !isResultZB && !isResultB), mbAs.size(), azCount, zSize);

		ArrayList<MatrixBlock> fuseInputs = new ArrayList<>();
		fuseInputs.addAll(mbABs);
		if(!isResultBZ && !isResultZB && !isResultB)
			fuseInputs.addAll(mbBs);
		fuseInputs.addAll(mbAs);
		if (isResultZ || isResultBZ || isResultZB)
			fuseInputs.addAll(mbAZs);

		ArrayList<ScalarObject> scalarObjects = new ArrayList<>();
		if(scalar != null){
			scalarObjects.add(new DoubleObject(scalar));
		}
		MatrixBlock out = r.execute(fuseInputs, scalarObjects, new MatrixBlock(), numThreads);

		if(isResultB && !mbBs.isEmpty()){
			LibMatrixMult.vectMultiply(mbBs.get(0).getDenseBlockValues(), out.getDenseBlockValues(), 0,0, mbABs.get(0).getNumColumns());
		}
		if(isResultBZ && !mbBs.isEmpty()){
			ensureMatrixBlockColumnVector(mbBs.get(0));
			out = out.binaryOperations(new BinaryOperator(Multiply.getMultiplyFnObject()), mbBs.get(0));
		}
		if(isResultZB && !mbBs.isEmpty()){
			ensureMatrixBlockRowVector(mbBs.get(0));
			out = out.binaryOperations(new BinaryOperator(Multiply.getMultiplyFnObject()), mbBs.get(0));
		}

		if( isResultA ||  isResultB || isResultZ)
			ensureMatrixBlockColumnVector(out);

		return out;
	}
    @Override
    public MatrixBlock computeEOpNode(ArrayList<MatrixBlock> inputs, int numThreads, Log LOG) {
		final Function<EOpNode, MatrixBlock> eOpNodeToMatrixBlock =  n -> n.computeEOpNode(inputs, numThreads, LOG);
        ArrayList<MatrixBlock> mbABs = new ArrayList<>(ABs.stream().map(eOpNodeToMatrixBlock).toList());
		List<MatrixBlock> mbBAs = BAs.stream().map(eOpNodeToMatrixBlock).toList();
		List<MatrixBlock> mbBs =  Bs.stream().map(eOpNodeToMatrixBlock).toList();
		List<MatrixBlock> mbAs = As.stream().map(eOpNodeToMatrixBlock).toList();
		List<MatrixBlock> mbAZs = AZs.stream().map(eOpNodeToMatrixBlock).toList();
		Double scalar = this.scalar == null ? null : this.scalar.computeEOpNode(inputs, numThreads, LOG).get(0,0);
		return EOpNodeFuse.compute(this.einsumRewriteType, mbABs, mbBAs, mbBs, mbAs, mbAZs , scalar, numThreads);
    }

    @Override
	public EOpNode reorderChildrenAndOptimize(EOpNode parent, Character outChar1, Character outChar2) {
		ABs.replaceAll(n -> n.reorderChildrenAndOptimize(this, n.c1, n.c2));
		BAs.replaceAll(n -> n.reorderChildrenAndOptimize(this, n.c1, n.c2));
		As.replaceAll(n -> n.reorderChildrenAndOptimize(this, n.c1, n.c2));
		Bs.replaceAll(n -> n.reorderChildrenAndOptimize(this, n.c1, n.c2));
		AZs.replaceAll(n -> n.reorderChildrenAndOptimize(this, n.c1, n.c2));
		return this;
    }

    private static @NotNull List<MatrixBlock> multiplyVectorsIntoOne(List<MatrixBlock> mbs, int size) {
        MatrixBlock mb = new MatrixBlock(mbs.get(0).getNumRows(), mbs.get(0).getNumColumns(), false);
        mb.allocateDenseBlock();
        for(int i = 1; i< mbs.size(); i++) { // multiply Bs
            if(i==1)
                LibMatrixMult.vectMultiplyWrite(mbs.get(0).getDenseBlock().values(0), mbs.get(1).getDenseBlock().values(0), mb.getDenseBlock().values(0),0,0,0, size);
            else
                LibMatrixMult.vectMultiply(mbs.get(i).getDenseBlock().values(0),mb.getDenseBlock().values(0),0,0, size);
        }
        return List.of(mb);
    }
}

