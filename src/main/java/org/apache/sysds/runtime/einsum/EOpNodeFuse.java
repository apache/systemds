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
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

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
//	public List<EOpNode> Zs;
//    public final List<List<EOpNode>> operands;
	public List<EOpNode> getAllOps(){
		List<EOpNode> all = new ArrayList<>();
		all.addAll(ABs);
		all.addAll(BAs);
		all.addAll(Bs);
		all.addAll(As);
		all.addAll(AZs);
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
//		this.Zs = Zs;
    }
	public EOpNodeFuse(EinsumRewriteType einsumRewriteType, List<EOpNode> ABs, List<EOpNode> BAs, List<EOpNode> Bs, List<EOpNode> As, List<EOpNode> AZs) {
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
//		this.Zs = Zs;
//		this.operands = Arrays.asList(operands);
	}
	@Override
	public String[] recursivePrintString() {
		ArrayList<String[]> inpStrings = new ArrayList<>();
		for (EOpNode node : getAllOps()) {
			inpStrings.add(node.recursivePrintString());
		}
		String[] inpRes = inpStrings.stream().flatMap(Arrays::stream).toArray(String[]::new);
		String[] scalarRes = this.scalar==null ? new String[]{} : this.scalar.recursivePrintString();
		String[] res = new String[1 + inpRes.length + scalarRes.length];

		res[0] = this.getClass().getSimpleName()+" ("+einsumRewriteType.toString()+") "+this.toString();

		for  (int i=0; i<inpRes.length; i++) {
			res[i+1] = (i==0 ?  "┌  " : (i==inpRes.length-1 ?  "└  " : "|  "))+inpRes[i];
		}
		for  (int i=0; i<scalarRes.length; i++) {
			res[i+inpRes.length+1] = (i==0 ?  "┌  " : (i==scalarRes.length-1 ?  "└  " : "|  "))+scalarRes[i];
		}
		return res;
	}
	public void addScalarAsIntermediate(EOpNode scalar) {
		if(einsumRewriteType == EinsumRewriteType.AB_BA_B_A__A || einsumRewriteType == EinsumRewriteType.AB_BA_B_A_AZ__Z)
			this.scalar = scalar;
		else
			throw new RuntimeException("EOpNodeFuse.addScalarAsIntermediate: scalar is undefined for type "+einsumRewriteType.toString());
	}

    public static List<EOpNodeFuse> findFuseOps(ArrayList<EOpNode> operands, Character outChar1, Character outChar2,/*, Set<Character> simplySummableChars,*/
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
		ArrayList<EOpNode> Zs = new ArrayList<>();

		HashSet<String> usedMatricesChars = new HashSet<>();
		HashSet<EOpNode> usedOperands = new HashSet<>();

		for(String ABCandidate : matricesCharsSorted.stream().map(Pair::getRight).toList()) {
			if(usedMatricesChars.contains(ABCandidate)) continue;

			char a = ABCandidate.charAt(0);
			char b = ABCandidate.charAt(1);
			String AB = ABCandidate;
			String BA = "" + b + a;

			int BAsCounter = (charsToMatrices.containsKey(BA) ? charsToMatrices.get(BA).size() : 0);
			int ABsCounter = charsToMatrices.get(AB).size();

			if(BAsCounter > ABsCounter + 1) {
				BA = "" + a + b;
				AB = "" + b + a;
				char tmp = a;
				a = b;
				b = tmp;
				int tmp2 = ABsCounter;
				ABsCounter = BAsCounter;
				BAsCounter = tmp2;
			}
			String A = "" + a;
			String B = "" + b;
			ArrayList<EOpNode> Bs = !charsToMatrices.containsKey(B) || usedMatricesChars.contains(B) ? new ArrayList<>() : charsToMatrices.get(B);
			ArrayList<EOpNode> As = !charsToMatrices.containsKey(A) || usedMatricesChars.contains(A) ? new ArrayList<>() : charsToMatrices.get(A);
			int AsCounter = As.size();
			int BsCounter = Bs.size();

			if(AsCounter == 0 && BsCounter == 0 && (ABsCounter + BAsCounter) < 2) { // no elementwise multiplication possible
				continue;
			}

			int usedBsCount = BsCounter + ABsCounter + BAsCounter;

			boolean doSumA = false;
			boolean doSumB = charToOccurences.get(b) == usedBsCount && (outChar1 == null || b != outChar1) && (outChar2 == null || b != outChar2);
//			boolean doSumZ = false; // there could be multiple AZ-s if Z is summed but for now it is limited to one
			HashSet<String> AZCandidates = matricesCharsStartingWithChar.get(a);
			boolean includeAZ = AZCandidates.size() == 2; // 2 because it also contains AB

			String AZ = null;
			Character z = null;
			if(includeAZ) {
				var it = AZCandidates.iterator(); AZ = it.next();
				if(AZ.charAt(1) == b) AZ = it.next(); // AB was chosen instead of AZ
				AZs = charsToMatrices.get(AZ);
				z = AZ.charAt(1);
//				String Z = "" + z;
//				Zs = charsToMatrices.get(Z);
				if(usedMatricesChars.contains(AZ)) { includeAZ = false; }
				int AZsCounter = AZs.size();
				doSumA = charToOccurences.get(a) == AsCounter + ABsCounter + BAsCounter + AZsCounter && (outChar1 == null || a != outChar1) && (outChar2 == null || a != outChar2);
//				doSumZ = charToOccurences.get(z) == AZsCounter + Zs.size();
				if(!doSumA) {
					includeAZ = false;
				}
				else if(!doSumB) { // check if outer is possible AB,...,AZ->BZ
					if(!EinsumCPInstruction.FUSE_OUTER_MULTIPLY
						|| (EinsumCPInstruction.FUSE_OUTER_MULTIPLY_EXCEEDS_L2_CACHE_CHECK && ((charToSize.get(a) * charToSize.get(b) *(ABsCounter + BAsCounter)) + (charToSize.get(a)*charToSize.get(z)*(AZsCounter))) * 8 < 6 * 1024 * 1024)
						|| !LibMatrixMult.isSkinnyRightHandSide(charToSize.get(AB.charAt(0)), charToSize.get(AB.charAt(1)),
							charToSize.get(AB.charAt(0)), charToSize.get(AZCandidates.iterator().next().charAt(1)),
							false)) {
						includeAZ = false;
					}
				}
				// else AB,...,AZ-> Z possible
			}

			if(!includeAZ) {
				doSumA = charToOccurences.get(a) == AsCounter + ABsCounter + BAsCounter && (outChar1 == null || a != outChar1) && (outChar2 == null || a != outChar2);
			}

			ArrayList<EOpNode> ABs = charsToMatrices.containsKey(AB) ? charsToMatrices.get(AB) : new ArrayList<>();
			ArrayList<EOpNode> BAs = charsToMatrices.containsKey(BA) ? charsToMatrices.get(BA) : new ArrayList<>();

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

//			if(type == EinsumRewriteType.AB_BA_B_A_AZ__Z && AZs.size() > 1){ // multiply all AZs if multiple
//				EOpNodeFuse fuseAZs = new EOpNodeFuse(EinsumRewriteType.AB_BA_B_A__AB, new ArrayList<>(AZs), new ArrayList<>(), new ArrayList<>(), new ArrayList<>(), new ArrayList<>());
//				AZs = new ArrayList<>();
//				AZs.add(fuseAZs);
//			}

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
            }else{
				charToOccurences.put(n.c1, charToOccurences.get(n.c1) - 1);
				if(charToOccurences.get(n.c2)!= null)
					charToOccurences.put(n.c2, charToOccurences.get(n.c2)-1);
			}
        }

        return result;
    }
	public static MatrixBlock compute(EinsumRewriteType rewriteType, List<MatrixBlock> ABsInput, List<MatrixBlock> mbBAs, List<MatrixBlock> mbBs, List<MatrixBlock> mbAs, List<MatrixBlock> mbAZs, Double scalar, int numThreads){
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
		//        int zCount = 0;
		switch(rewriteType){
			case AB_BA_B_A_AZ__Z ->  {
				constDim2 = mbAZs.get(0).getNumColumns();
				zSize = mbAZs.get(0).getNumColumns();
				azCount = mbAZs.size();
				//                if (mbZs != null) zCount = mbZs.size();
			}
			case AB_BA_A_AZ__BZ, AB_BA_A_AZ__ZB -> {
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
        ArrayList<MatrixBlock> mbABs = new ArrayList<>(ABs.stream().map(n -> n.computeEOpNode(inputs, numThreads, LOG)).toList());
		List<MatrixBlock> mbBAs = BAs.stream().map(n -> n.computeEOpNode(inputs, numThreads, LOG)).toList();
		List<MatrixBlock> mbBs =  Bs.stream().map(n -> n.computeEOpNode(inputs, numThreads, LOG)).toList();
		List<MatrixBlock> mbAs = As.stream().map(n -> n.computeEOpNode(inputs, numThreads, LOG)).toList();
        List<MatrixBlock> mbAZs = AZs.stream().map(n -> n.computeEOpNode(inputs, numThreads, LOG)).toList();
		Double scalar = this.scalar == null ? null : this.scalar.computeEOpNode(inputs, numThreads, LOG).get(0,0);
		return EOpNodeFuse.compute(this.einsumRewriteType, mbABs, mbBAs, mbBs, mbAs, mbAZs, scalar, numThreads);
    }

    @Override
	public EOpNode reorderChildrenAndOptimize(EOpNode parent, Character outChar1, Character outChar2) {
		for(int i = 0; i < ABs.size(); i++) ABs.set(i,ABs.get(i).reorderChildrenAndOptimize(this, ABs.get(i).c1, ABs.get(i).c2));
		for(int i = 0; i < BAs.size(); i++) BAs.set(i,BAs.get(i).reorderChildrenAndOptimize(this, BAs.get(i).c1, BAs.get(i).c2));
		for(int i = 0; i < As.size(); i++) As.set(i,As.get(i).reorderChildrenAndOptimize(this, As.get(i).c1, As.get(i).c2));
		for(int i = 0; i < Bs.size(); i++) Bs.set(i,Bs.get(i).reorderChildrenAndOptimize(this, Bs.get(i).c1, Bs.get(i).c2));
		for(int i = 0; i < AZs.size(); i++) AZs.set(i,AZs.get(i).reorderChildrenAndOptimize(this, AZs.get(i).c1, AZs.get(i).c2));
		return this;
    }

    private static @NotNull List<MatrixBlock> multiplyVectorsIntoOne(List<MatrixBlock> mbs, int size) {
        MatrixBlock mb = new MatrixBlock(mbs.get(0).getNumRows(), mbs.get(0).getNumColumns(), false);
        mb.allocateDenseBlock();
        for(int i = 1; i< mbs.size(); i++) { // multiply Bs
            if(i==1){
                LibMatrixMult.vectMultiplyWrite(mbs.get(0).getDenseBlock().values(0), mbs.get(1).getDenseBlock().values(0), mb.getDenseBlock().values(0),0,0,0, size);
            }else{
                LibMatrixMult.vectMultiply(mbs.get(i).getDenseBlock().values(0),mb.getDenseBlock().values(0),0,0, size);
            }
        }
        return List.of(mb);
    }
}

