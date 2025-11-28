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
import org.apache.commons.lang3.tuple.Triple;
import org.apache.commons.logging.Log;
import org.apache.sysds.runtime.codegen.LibSpoofPrimitives;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.functionobjects.SwapIndex;
import org.apache.sysds.runtime.instructions.cp.DoubleObject;
import org.apache.sysds.runtime.instructions.cp.EinsumCPInstruction;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.matrix.data.LibMatrixMult;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.AggregateOperator;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.ReorgOperator;
import org.apache.sysds.runtime.matrix.operators.SimpleOperator;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.function.Predicate;

import static org.apache.sysds.runtime.instructions.cp.EinsumCPInstruction.ensureMatrixBlockColumnVector;
import static org.apache.sysds.runtime.instructions.cp.EinsumCPInstruction.ensureMatrixBlockRowVector;

public class EOpNodeBinary extends EOpNode {

	public enum EBinaryOperand { // upper case: char has to remain, lower case: to be summed
        ////// mm:   //////
        Ba_aC, // -> BC
        aB_Ca, // -> CB
        Ba_Ca, // -> BC
        aB_aC, // -> BC

        ////// elementwisemult and sums //////
        aB_aB,// elemwise and colsum -> B
		Ab_Ab, // elemwise and rowsum ->A
		Ab_bA, // elemwise, either colsum or rowsum -> A
		aB_Ba,
		ab_ab,//M-M sum all
		ab_ba, //M-M.T sum all
		aB_a,// -> B
		Ab_b, // -> A

        ////// elementwise, no summations:   //////
        A_A,// v-elemwise -> A
        AB_AB,// M-M elemwise -> AB
        AB_BA, // M-M.T elemwise -> AB
        AB_A, // M-v colwise -> BA!?
		AB_B, // M-v rowwise -> AB

        ////// other   //////
		a_a,// dot ->
        A_B, // outer mult -> AB
        A_scalar, // v-scalar
        AB_scalar, // m-scalar
        scalar_scalar
    }
    public EOpNode left;
    public EOpNode right;
    public EBinaryOperand operand;
	private boolean transposeResult;
	public EOpNodeBinary(EOpNode left, EOpNode right, EBinaryOperand operand){
		super(null,null,null, null);
		Character c1, c2;
		Integer dim1, dim2;
		switch(operand){
			case Ba_aC -> {
				c1=left.c1;
				c2=right.c2;
				dim1=left.dim1;
				dim2=right.dim2;
			}
			case aB_Ca -> {
				c1=left.c2;
				c2=right.c1;
				dim1=left.dim2;
				dim2=right.dim1;
			}
			case Ba_Ca -> {
				c1=left.c1;
				c2=right.c1;
				dim1=left.dim1;
				dim2=right.dim1;
			}
			case aB_aC -> {
				c1=left.c2;
				c2=right.c2;
				dim1=left.dim2;
				dim2=right.dim2;
			}
			case aB_aB, aB_Ba, aB_a -> {
				c1=left.c2;
				c2=null;
				dim1=left.dim2;
				dim2=null;
			}
			case Ab_Ab, Ab_bA, Ab_b, A_A, A_scalar -> {
				c1=left.c1;
				c2=null;
				dim1=left.dim1;
				dim2=null;
			}
			case ab_ab, ab_ba, a_a, scalar_scalar -> {
				c1=null;
				c2=null;
				dim1=null;
				dim2=null;
			}
			case AB_AB, AB_BA, AB_A, AB_B, AB_scalar ->{
				c1=left.c1;
				c2=left.c2;
				dim1=left.dim1;
				dim2=left.dim2;
			}
			case A_B -> {
				c1=left.c1;
				c2=right.c1;
				dim1=left.dim1;
				dim2=right.dim1;
			}
			default -> throw new IllegalStateException("EOpNodeBinary Unexpected type: " + operand);
		}
		//	super(c1, c2, dim1, dim2); // unavailable in JDK < 22
		this.c1 = c1;
		this.c2 = c2;
		this.dim1 = dim1;
		this.dim2 = dim2;
		this.left = left;
		this.right = right;
		this.operand = operand;
	}

	public void setTransposeResult(boolean transposeResult){
		this.transposeResult = transposeResult;
	}

	public static EOpNodeBinary combineMatrixMultiply(EOpNode left, EOpNode right) {
		if (left.c2 == right.c1) { return new EOpNodeBinary(left, right, EBinaryOperand.Ba_aC); }
		if (left.c2 == right.c2) { return new EOpNodeBinary(left, right, EBinaryOperand.Ba_Ca); }
		if (left.c1 == right.c1) { return new EOpNodeBinary(left, right, EBinaryOperand.aB_aC); }
		if (left.c1 == right.c2) {
			var res = new EOpNodeBinary(left, right, EBinaryOperand.aB_Ca);
			res.setTransposeResult(true);
			return res;
		}
		throw new RuntimeException("EOpNodeBinary::combineMatrixMultiply: invalid matrix operation");
	}

	@Override
	public String[] recursivePrintString() {
		String[] left = this.left.recursivePrintString();
		String[] right = this.right.recursivePrintString();
		String[] res = new String[left.length + right.length+1];
		res[0] = this.getClass().getSimpleName()+" ("+ operand.toString()+") "+this.toString();
		for (int i=0; i<left.length; i++) {
			res[i+1] = (i==0 ?  "┌─ " : "   ") +left[i];
		}
		for (int i=0; i<right.length; i++) {
			res[left.length+i+1] = (i==0 ?  "└─ " : "   ") +right[i];
		}
		return res;
	}

	@Override
    public MatrixBlock computeEOpNode(ArrayList<MatrixBlock> inputs, int numThreads, Log LOG) {
        EOpNodeBinary bin = this;
        MatrixBlock left = this.left.computeEOpNode(inputs, numThreads, LOG);
        MatrixBlock right = this.right.computeEOpNode(inputs, numThreads, LOG);

        AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());

        MatrixBlock res;

        if(LOG.isTraceEnabled()) LOG.trace("computing binary "+bin.left +","+bin.right +"->"+bin);

        switch (bin.operand){
            case AB_AB -> {
                res = MatrixBlock.naryOperations(new SimpleOperator(Multiply.getMultiplyFnObject()), new MatrixBlock[]{left, right},new ScalarObject[]{}, new MatrixBlock());
            }
            case A_A -> {
                ensureMatrixBlockColumnVector(left);
                ensureMatrixBlockColumnVector(right);
                res = MatrixBlock.naryOperations(new SimpleOperator(Multiply.getMultiplyFnObject()), new MatrixBlock[]{left, right},new ScalarObject[]{}, new MatrixBlock());
            }
            case a_a -> {
                ensureMatrixBlockColumnVector(left);
                ensureMatrixBlockColumnVector(right);
				res = new MatrixBlock(0.0);
				res.allocateDenseBlock();
				res.getDenseBlockValues()[0] = LibMatrixMult.dotProduct(left.getDenseBlockValues(), right.getDenseBlockValues(), 0,0 , left.getNumRows());
            }
            case Ab_Ab -> {
				res = EOpNodeFuse.compute(EOpNodeFuse.EinsumRewriteType.AB_BA_B_A__A, List.of(left, right), new ArrayList<>(), new ArrayList<>(), new ArrayList<>(), new ArrayList<>(),null,numThreads);
			}
            case aB_aB -> {
				res = EOpNodeFuse.compute(EOpNodeFuse.EinsumRewriteType.AB_BA_A__B, List.of(left, right), new ArrayList<>(), new ArrayList<>(), new ArrayList<>(), new ArrayList<>(),null,numThreads);
            }
            case ab_ab -> {
				res = EOpNodeFuse.compute(EOpNodeFuse.EinsumRewriteType.AB_BA_B_A__, List.of(left, right), new ArrayList<>(), new ArrayList<>(), new ArrayList<>(), new ArrayList<>(),null,numThreads);
			}
            case ab_ba -> {
				res = EOpNodeFuse.compute(EOpNodeFuse.EinsumRewriteType.AB_BA_B_A__, List.of(left), List.of(right), new ArrayList<>(), new ArrayList<>(), new ArrayList<>(),null,numThreads);
			}
            case Ab_bA -> {
				res = EOpNodeFuse.compute(EOpNodeFuse.EinsumRewriteType.AB_BA_B_A__A, List.of(left), List.of(right), new ArrayList<>(), new ArrayList<>(), new ArrayList<>(),null,numThreads);
			}
            case aB_Ba -> {
				res = EOpNodeFuse.compute(EOpNodeFuse.EinsumRewriteType.AB_BA_A__B, List.of(left), List.of(right), new ArrayList<>(), new ArrayList<>(), new ArrayList<>(),null,numThreads);
			}
            case AB_BA -> {
                ReorgOperator transpose = new ReorgOperator(SwapIndex.getSwapIndexFnObject(), numThreads);
                right = right.reorgOperations(transpose, new MatrixBlock(), 0, 0, 0);
                res = MatrixBlock.naryOperations(new SimpleOperator(Multiply.getMultiplyFnObject()), new MatrixBlock[]{left, right},new ScalarObject[]{}, new MatrixBlock());
            }
            case Ba_aC -> {
                res = LibMatrixMult.matrixMult(left,right, new MatrixBlock(), numThreads);
            }
            case aB_Ca -> {
                res = LibMatrixMult.matrixMult(right,left, new MatrixBlock(), numThreads);
            }
            case Ba_Ca -> {
                ReorgOperator transpose = new ReorgOperator(SwapIndex.getSwapIndexFnObject(), numThreads);
                right = right.reorgOperations(transpose, new MatrixBlock(), 0, 0, 0);
                res = LibMatrixMult.matrixMult(left,right, new MatrixBlock(), numThreads);
            }
            case aB_aC -> {
                if(false && LibMatrixMult.isSkinnyRightHandSide(left.getNumRows(), left.getNumColumns(), right.getNumRows(), right.getNumColumns(), false)){
                    res = new MatrixBlock(left.getNumColumns(), right.getNumColumns(),false);
                    res.allocateDenseBlock();
                    double[] m1 = left.getDenseBlock().values(0);
                    double[] m2 = right.getDenseBlock().values(0);
                    double[] c = res.getDenseBlock().values(0);
                    int alen = left.getNumColumns();
                    int blen = right.getNumColumns();
                    for(int i =0;i<left.getNumRows();i++){
                        LibSpoofPrimitives.vectOuterMultAdd(m1,m2,c,i*alen,i*blen, 0,alen,blen);
                    }
                }else {
                    ReorgOperator transpose = new ReorgOperator(SwapIndex.getSwapIndexFnObject(), numThreads);
                    left = left.reorgOperations(transpose, new MatrixBlock(), 0, 0, 0);
                    res = LibMatrixMult.matrixMult(left, right, new MatrixBlock(), numThreads);
                }
            }
            case A_scalar, AB_scalar -> {
                res = MatrixBlock.naryOperations(new SimpleOperator(Multiply.getMultiplyFnObject()), new MatrixBlock[]{left},new ScalarObject[]{new DoubleObject(right.get(0,0))}, new MatrixBlock());
            }
            case AB_B -> {
                ensureMatrixBlockRowVector(right);
                res = left.binaryOperations(new BinaryOperator(Multiply.getMultiplyFnObject()), right);
            }
            case Ab_b -> {
				res = EOpNodeFuse.compute(EOpNodeFuse.EinsumRewriteType.AB_BA_B_A__A, List.of(left), new ArrayList<>(), List.of(right), new ArrayList<>(), new ArrayList<>(),null,numThreads);
			}
            case AB_A -> {
                ensureMatrixBlockColumnVector(right);
                res = left.binaryOperations(new BinaryOperator(Multiply.getMultiplyFnObject()), right);
            }
            case aB_a -> {
				res = EOpNodeFuse.compute(EOpNodeFuse.EinsumRewriteType.AB_BA_A__B, List.of(left), new ArrayList<>(), new ArrayList<>(), List.of(right), new ArrayList<>(),null,numThreads);
			}
            case A_B -> {
                ensureMatrixBlockColumnVector(left);
                ensureMatrixBlockRowVector(right);
                res = left.binaryOperations(new BinaryOperator(Multiply.getMultiplyFnObject()), right);
            }
            case scalar_scalar -> {
                return new MatrixBlock(left.get(0,0)*right.get(0,0));
            }
            default -> {
                throw new IllegalArgumentException("Unexpected value: " + bin.operand.toString());
            }

        }
		if(transposeResult){
			ReorgOperator transpose = new ReorgOperator(SwapIndex.getSwapIndexFnObject(), numThreads);
			res = res.reorgOperations(transpose, new MatrixBlock(), 0, 0, 0);
		}
		if(c2 == null) ensureMatrixBlockColumnVector(res);
        return res;
    }

    @Override
    public EOpNode reorderChildrenAndOptimize(EOpNode parent, Character outChar1, Character outChar2) {
        if (this.operand ==EBinaryOperand.aB_aC){
            if(this.right.c2 == outChar1) { // result is CB so Swap aB and aC
				var tmpLeft = left;  left = right;  right = tmpLeft;
				var tmpC1 = c1;       c1 = c2;         c2 = tmpC1;
				var tmpDim1 = dim1;   dim1 = dim2;     dim2 = tmpDim1;
            }
			if(EinsumCPInstruction.FUSE_OUTER_MULTIPLY && left instanceof EOpNodeFuse fuse && fuse.einsumRewriteType == EOpNodeFuse.EinsumRewriteType.AB_BA_B_A__AB
				&& (!EinsumCPInstruction.FUSE_OUTER_MULTIPLY_EXCEEDS_L2_CACHE_CHECK || ((fuse.dim1 * fuse.dim2 *(fuse.ABs.size()+fuse.BAs.size())) + (right.dim1*right.dim2)) * 8 > 6 * 1024 * 1024)
				&& LibMatrixMult.isSkinnyRightHandSide(left.dim1, left.dim2,  right.dim1, right.dim2, false)) {
				fuse.AZs.add(right);
				fuse.einsumRewriteType = EOpNodeFuse.EinsumRewriteType.AB_BA_A_AZ__BZ;
				fuse.c1 = fuse.c2;
				fuse.c2 = right.c2;
				return fuse;
			}

            left = left.reorderChildrenAndOptimize(this, left.c2, left.c1); // maybe can be reordered
            if(left.c2 == right.c1) { // check if change happened:
                this.operand = EBinaryOperand.Ba_aC;
            }
			right =  right.reorderChildrenAndOptimize(this, right.c1, right.c2);
        }else if (this.operand ==EBinaryOperand.Ba_Ca){
			if(this.right.c1 == outChar1) { // result is CB so Swap Ba and Ca
				var tmpLeft = left;  left = right;  right = tmpLeft;
				var tmpC1 = c1;       c1 = c2;         c2 = tmpC1;
				var tmpDim1 = dim1;   dim1 = dim2;     dim2 = tmpDim1;
			}

			right = right.reorderChildrenAndOptimize(this, right.c2, right.c1); // maybe can be reordered
			if(left.c2 == right.c1) { // check if change happened:
				this.operand = EBinaryOperand.Ba_aC;
			}
			left = left.reorderChildrenAndOptimize(this, left.c1, left.c2);
		}else {
			left = left.reorderChildrenAndOptimize(this, left.c1, left.c2); // just recurse
			right = right.reorderChildrenAndOptimize(this, right.c1, right.c2);
		}
		return this;
    }

	// used in the old approach
	public static Triple<Integer, EBinaryOperand, Pair<Character, Character>> TryCombineAndCost(EOpNode n1 , EOpNode n2, HashMap<Character, Integer> charToSizeMap, HashMap<Character, Integer> charToOccurences, Character outChar1, Character outChar2){
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
						return Triple.of(charToSizeMap.get(n1.c1)*charToSizeMap.get(n1.c2), EBinaryOperand.Ab_Ab, Pair.of(n1.c1, null));
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
						return Triple.of(charToSizeMap.get(n1.c1)*charToSizeMap.get(n2.c1), EBinaryOperand.AB_B, Pair.of(n1.c1, n1.c2));
					}
					return Triple.of(charToSizeMap.get(n1.c1)*charToSizeMap.get(n2.c1), EBinaryOperand.Ab_b, Pair.of(n1.c1, null));
				}
				return null; // AB,C
			}
			else if (n1.c2 == n2.c1) {
				if(n1.c1 == n2.c2){ // ab,ba
					if(cannotBeSummed.test(n1.c1)){
						if(cannotBeSummed.test(n1.c2)){
							return Triple.of(charToSizeMap.get(n1.c1)*charToSizeMap.get(n1.c2), EBinaryOperand.AB_BA, Pair.of(n1.c1, n1.c2));
						}
						return Triple.of(charToSizeMap.get(n1.c1)*charToSizeMap.get(n1.c2), EBinaryOperand.Ab_bA, Pair.of(n1.c1, null));
					}
					if(cannotBeSummed.test(n1.c2)){
						return Triple.of(charToSizeMap.get(n1.c1)*charToSizeMap.get(n1.c2), EBinaryOperand.aB_Ba, Pair.of(n1.c2, null));
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
			else { // something like AB,CD
				return null;
			}
		}
	}
}
