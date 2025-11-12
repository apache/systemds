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

import org.apache.commons.logging.Log;
import org.apache.sysds.common.Types;
import org.apache.sysds.hops.LiteralOp;
import org.apache.sysds.hops.codegen.cplan.CNode;
import org.apache.sysds.hops.codegen.cplan.CNodeData;
import org.apache.sysds.hops.codegen.cplan.CNodeRow;
import org.apache.sysds.runtime.codegen.CodegenUtils;
import org.apache.sysds.runtime.codegen.SpoofOperator;
import org.apache.sysds.runtime.codegen.SpoofRowwise;
import org.apache.sysds.runtime.controlprogram.parfor.util.IDSequence;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.functionobjects.ReduceCol;
import org.apache.sysds.runtime.functionobjects.ReduceRow;
import org.apache.sysds.runtime.functionobjects.SwapIndex;
import org.apache.sysds.runtime.instructions.cp.DoubleObject;
import org.apache.sysds.runtime.instructions.cp.EinsumCPInstruction;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.matrix.data.LibMatrixMult;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.AggregateOperator;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysds.runtime.matrix.operators.ReorgOperator;
import org.jetbrains.annotations.NotNull;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.function.Supplier;
import java.util.stream.Collectors;

import static org.apache.sysds.runtime.instructions.cp.EinsumCPInstruction.ensureMatrixBlockColumnVector;

public class EOpNodeFuse extends EOpNode {

	private EOpNode scalar = null;



	public enum EinsumRewriteType{
        // B -> row*vec, A -> row*scalar
        AB_BA_B_A__AB,
        AB_BA_B_A__B,
        AB_BA_B_A__A,
        AB_BA_B_A__,

        // scalar from row(AB).dot(B) multiplied by row(AZ)
        AB_BA_B_A_AZ__Z,

        // AZ: last step is outer matrix multiplication using vector Z
        AB_BA_B_A_AZ__BZ,
        AB_BA_B_A_AZ__ZB,
    }

    public final EinsumRewriteType einsumRewriteType;
    public final List<List<EOpNode>> operands;

    private EOpNodeFuse(Character c1, Character c2, EinsumRewriteType einsumRewriteType, List<EOpNode>... operands) {
        super(c1,c2);
        this.einsumRewriteType = einsumRewriteType;
        this.operands = Arrays.asList(operands);
    }
	@Override
	public String[] recursivePrintString() {
		ArrayList<String[]> inpStrings = new ArrayList<>();
		for (List<EOpNode> list : operands) {
			for (EOpNode node : list) {
				inpStrings.add(node.recursivePrintString());
			}
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

    public static EOpNodeFuse match(ArrayList<EOpNode> operands, Character outChar1, Character outChar2,/*, Set<Character> simplySummableChars,*/
		HashMap<Character, Integer> charToSize, HashMap<Character, Integer> charToOccurences, ArrayList<EOpNode> ret){
        //precompute
        HashSet<String> matricesChars = new HashSet<>();
        HashMap<String, ArrayList<EOpNode>> charsToMatrices = new HashMap<>();

        for (EOpNode operand1 : operands) {
            String k;

            if (operand1.c2 != null) {
                k = operand1.c1.toString() + operand1.c2;
                matricesChars.add(k);
            } else {
                k = operand1.c1.toString();
            }

            if (charsToMatrices.containsKey(k)) {
                charsToMatrices.get(k).add(operand1);
            } else {
                ArrayList<EOpNode> matrices = new ArrayList<>();
                matrices.add(operand1);
                charsToMatrices.put(k, matrices);
            }
        }

        ArrayList<EOpNode> AZs = new ArrayList<>();
        ArrayList<EOpNode> Zs = new ArrayList<>();
        boolean pass = false;

        String AB = null;
        String BA = null;
        boolean doSumA=false;
        boolean doSumB=false;
        for (String ABcandidate : matricesChars) {
            char a = ABcandidate.charAt(0);
            char b = ABcandidate.charAt(1);
            BA = "" + b + a;

            AZs = new ArrayList<>();
            Character z = null;
            pass=true;
            int AZsCounter = 0;
            HashSet<String> AZCandidates = new HashSet<>();

            for (String chars : charsToMatrices.keySet()) {
                if (chars.equals(ABcandidate) || chars.equals(BA)) {
                    continue;
                }

                if(chars.length()==1){
                    //always ok
                }else{
                    if(a==chars.charAt(1) && b==chars.charAt(0)){ //BA
                        continue;
                    }
                    if(chars.charAt(0)==a){ //AZ
                        AZsCounter++;
                        AZCandidates.add(chars);
                    }
                    else if(chars.charAt(0)==b){
                        // BZ
                    }
                    else if(chars.charAt(1)==a){
                        //ZA
                    }
                    else if(chars.charAt(1)==b){
                        // ZB
                    }
                }
            }

            if(pass){ // final checks for current AB candidate

                AB = ABcandidate;
                String A = ""+a;
                String B = ""+b;
                int BAsCounter = (charsToMatrices.containsKey(BA) ?  charsToMatrices.get(BA).size() : 0);
                int ABsCounter = charsToMatrices.get(ABcandidate).size()+BAsCounter;
                int AsCounter = (charsToMatrices.containsKey(A) ?  charsToMatrices.get(A).size() : 0);
                int BsCounter = (charsToMatrices.containsKey(B) ?  charsToMatrices.get(B).size() : 0);
                if(AsCounter==0 && BsCounter==0 && ABsCounter<2){
                    pass=false;
                    continue;
                }
                int usedBsCount = BsCounter+ABsCounter;
                doSumB = charToOccurences.get(b)==usedBsCount && (outChar1 == null || b!=outChar1) && (outChar2 == null || b!=outChar2);

                boolean includeAz = true;
                if(AZCandidates.size()==1){
                    if(!doSumB) {
                        // check if outer is possible AB,...,AZ->BZ
                        if(!EinsumCPInstruction.FUSE_OUTER_MULTIPLY || !LibMatrixMult.isSkinnyRightHandSide(charToSize.get(AB.charAt(0)), charToSize.get(AB.charAt(1)),  charToSize.get(AB.charAt(0)),charToSize.get(AZCandidates.iterator().next().charAt(1)),false)) {
                            includeAz=false;
                        }
                    }
                    if(includeAz){
                        int usedAsCount = AsCounter+ABsCounter+AZsCounter;
                        doSumA = charToOccurences.get(a)==usedAsCount && (outChar1 == null || a!=outChar1) && (outChar2 == null || a!=outChar2);
                        if(!doSumA){ // cant do AZ
                            break;// just do AB,B,A ->AB / A
                        }else {
                            AZs = charsToMatrices.get(AZCandidates.iterator().next());
                            break;//ok
                        }
                    }
                }
//				else if (AZCandidates.size() >= 2) {
//                    doSumA = false;
//                    if(doSumB){
//                        pass=true;
//                        break; // can do it, it will create AB,B,A -> A, that will be consumed by some AZ later
//                    }
//                    pass=false;
//                    continue;
//
//                }
                int usedAsCount = AsCounter+ABsCounter;
                doSumA = charToOccurences.get(a)==usedAsCount && (outChar1 == null || a!=outChar1) && (outChar2 == null || a!=outChar2);

                break;
            }
        }

        if(!pass){
            return null;
        }
        ArrayList<EOpNode> ABs=charsToMatrices.containsKey(AB) ? charsToMatrices.get(AB) : new ArrayList<>();
        ArrayList<EOpNode> BAs=charsToMatrices.containsKey(BA) ? charsToMatrices.get(BA) : new ArrayList<>();
        if (ABs.size() < BAs.size() - 1) {
            String tmp = AB;

            AB=BA;
            BA=tmp;
            ArrayList<EOpNode> tmp2 = ABs;
            BAs=ABs;
            ABs=tmp2;
			AZs.clear();
        }
        String B = AB.substring(1,2);
        String A = AB.substring(0,1);
        char a = A.charAt(0);
        char b = B.charAt(0);
        Character c1 = null;
        Character c2 = null;
        EinsumRewriteType t = null;

        if(!AZs.isEmpty()){
            Character azC2 = AZs.get(0).c2;
            if(doSumB) {
                t = EinsumRewriteType.AB_BA_B_A_AZ__Z;
                c1 = azC2;
            }
            else if (EinsumCPInstruction.FUSE_OUTER_MULTIPLY) {
                if(LibMatrixMult.isSkinnyRightHandSide(charToSize.get(AB.charAt(0)), charToSize.get(AB.charAt(1)),  charToSize.get(AB.charAt(0)),charToSize.get(azC2),false)){
                    if (outChar1 == azC2 && outChar2 == b) {
                        t = EinsumRewriteType.AB_BA_B_A_AZ__ZB;
                        c1 = azC2;
                        c2 = b;
                    } else if (outChar2 == azC2 && outChar1 == b) {
                        t = EinsumRewriteType.AB_BA_B_A_AZ__BZ;
                        c1 = b;
                        c2 = azC2;
                    } else {
                        t = EinsumRewriteType.AB_BA_B_A_AZ__ZB;
                        c1 = azC2;
                        c2 = b;
                    }

                }else{
                    t=null;
                    AZs=new ArrayList<>();
                }
            }else{
                t=null;
                AZs=new ArrayList<>();
            }

            if(charsToMatrices.containsKey(azC2.toString())) {
                Zs = charsToMatrices.get(azC2.toString());
            }
        }
        if(t==null) {
            if (doSumA) {
                if (doSumB) {
                    t = EinsumRewriteType.AB_BA_B_A__;
                } else {
                    t = EinsumRewriteType.AB_BA_B_A__B;
                    c1 = AB.charAt(1);
                }
            } else if (doSumB) {
                t = EinsumRewriteType.AB_BA_B_A__A;
                c1 = AB.charAt(0);
            } else {
                t = EinsumRewriteType.AB_BA_B_A__AB;
                c1 = AB.charAt(0);
                c2 = AB.charAt(1);
            }
        }
        if(c1 != null){
            charToOccurences.put(c1, charToOccurences.get(c1)+1);
        }
        if(c2 != null){
            charToOccurences.put(c2, charToOccurences.get(c2)+1);
        }
        HashSet<EOpNode> usedOperands = new HashSet<>();


        ArrayList<EOpNode> Bs=charsToMatrices.containsKey(B) ? charsToMatrices.get(B) : new ArrayList<>();
        ArrayList<EOpNode> As=charsToMatrices.containsKey(A) ? charsToMatrices.get(A) : new ArrayList<>();

        usedOperands.addAll(ABs);
        usedOperands.addAll(BAs);
        usedOperands.addAll(Bs);
        usedOperands.addAll(As);
        usedOperands.addAll(AZs);
        usedOperands.addAll(Zs);

        for(EOpNode n : operands){
            if(!usedOperands.contains(n)){
                ret.add(n);
            }else{
                if(charToOccurences != null){
                    charToOccurences.put(n.c1, charToOccurences.get(n.c1)-1);
                    if(charToOccurences.get(n.c2)!= null)
                        charToOccurences.put(n.c2, charToOccurences.get(n.c2)-1);
                }
            }
        }

        var e = new EOpNodeFuse(c1, c2, t,
                ABs,
                BAs,
                Bs,
                As,
                AZs,
                Zs
        );
        return e;
    }

    @Override
    public MatrixBlock computeEOpNode(ArrayList<MatrixBlock> inputs, int numThreads, Log LOG) {
		List<List<MatrixBlock>> mbs = operands.stream().map(l -> l.stream().map(n -> n.computeEOpNode(inputs, numThreads, LOG)).collect(Collectors.toList())).toList();
        var eOpNodeEinsumFuse = this;

        if( LOG.isTraceEnabled()) {
            String x = eOpNodeEinsumFuse.operands.stream()
                    .flatMap(List::stream)
                    .map(o -> o.c1.toString() + (o.c2 == null ? "" : o.c2))
                    .collect(Collectors.joining(","));
            String res = eOpNodeEinsumFuse.c1 == null ? "" : (eOpNodeEinsumFuse.c1.toString() +(eOpNodeEinsumFuse.c2 == null ? "" : eOpNodeEinsumFuse.c2.toString()));

            LOG.trace("ComputeEOpNodeFuse AB=" + operands.get(0).get(0).c1+operands.get(0).get(0).c2 +" "+eOpNodeEinsumFuse.einsumRewriteType.toString() +" "+ x + "->" + res);
        }
        boolean isResultAB = eOpNodeEinsumFuse.einsumRewriteType == EOpNodeFuse.EinsumRewriteType.AB_BA_B_A__AB;
        boolean isResultA = eOpNodeEinsumFuse.einsumRewriteType == EOpNodeFuse.EinsumRewriteType.AB_BA_B_A__A;
        boolean isResultB = eOpNodeEinsumFuse.einsumRewriteType == EOpNodeFuse.EinsumRewriteType.AB_BA_B_A__B;
        boolean isResult_ = eOpNodeEinsumFuse.einsumRewriteType == EOpNodeFuse.EinsumRewriteType.AB_BA_B_A__;
        boolean isResultZ = eOpNodeEinsumFuse.einsumRewriteType == EOpNodeFuse.EinsumRewriteType.AB_BA_B_A_AZ__Z;
        boolean isResultBZ = eOpNodeEinsumFuse.einsumRewriteType == EOpNodeFuse.EinsumRewriteType.AB_BA_B_A_AZ__BZ;
        boolean isResultZB = eOpNodeEinsumFuse.einsumRewriteType == EOpNodeFuse.EinsumRewriteType.AB_BA_B_A_AZ__ZB;
        List<MatrixBlock> ABs = mbs.get(0), BAs = mbs.get(1), Bs =  mbs.get(2), As = mbs.get(3);
        List<MatrixBlock> AZs = mbs.get(4);
        List<MatrixBlock> Zs = mbs.get(5);
        int bSize = ABs.get(0).getNumColumns();
        int aSize = ABs.get(0).getNumRows();
		if (!BAs.isEmpty()) {
			ReorgOperator transpose = new ReorgOperator(SwapIndex.getSwapIndexFnObject(), numThreads);
			for(MatrixBlock mb : BAs) {//BA->AB
				ABs.add(mb.reorgOperations(transpose, null, 0, 0, 0));
			}
		}
        AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
        AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceRow.getReduceRowFnObject(), numThreads);

        if(As.size() > 1){
            As = multiplyVectorsIntoOne(As, aSize);
        }
        if(Bs.size() > 1){
            Bs = multiplyVectorsIntoOne(Bs, bSize);
        }
        if(Zs != null && Zs.size() > 1){
            Zs = multiplyVectorsIntoOne(Zs, AZs.get(0).getNumColumns());
        }
        int constDim2 = -1;
        int zSize = 0;
        int azCount = 0;
        int zCount = 0;
        switch(eOpNodeEinsumFuse.einsumRewriteType){
            case AB_BA_B_A_AZ__Z ->  {
                constDim2 = AZs.get(0).getNumColumns();
                zSize = AZs.get(0).getNumColumns();
                azCount = AZs.size();
                if (Zs != null) zCount = Zs.size();
            }
            case AB_BA_B_A_AZ__BZ, AB_BA_B_A_AZ__ZB -> {
                constDim2 = AZs.get(0).getNumColumns();
                zSize = AZs.get(0).getNumColumns();
                azCount = AZs.size();
            }
        }

        SpoofRowwise.RowType rowType = switch(eOpNodeEinsumFuse.einsumRewriteType){
            case AB_BA_B_A__AB -> SpoofRowwise.RowType.NO_AGG;
            case AB_BA_B_A__B -> SpoofRowwise.RowType.COL_AGG_T;
            case AB_BA_B_A__A -> SpoofRowwise.RowType.ROW_AGG;
            case AB_BA_B_A__ -> SpoofRowwise.RowType.FULL_AGG;
            case AB_BA_B_A_AZ__Z -> SpoofRowwise.RowType.COL_AGG_CONST;
            case AB_BA_B_A_AZ__BZ -> SpoofRowwise.RowType.COL_AGG_B1_T;
            case AB_BA_B_A_AZ__ZB -> SpoofRowwise.RowType.COL_AGG_B1;
        };
        EinsumSpoofRowwise r = new EinsumSpoofRowwise(eOpNodeEinsumFuse.einsumRewriteType, rowType, constDim2, false, 1, ABs.size()-1,Bs.size(), As.size(), zCount, azCount, zSize);


        ArrayList<MatrixBlock> fuseInputs = new ArrayList<>();

        fuseInputs.addAll(ABs);
        fuseInputs.addAll(Bs);
        fuseInputs.addAll(As);
        if (isResultZ || isResultBZ || isResultZB)
            fuseInputs.addAll(AZs);
		ArrayList<ScalarObject> scalarObjects = new ArrayList<>();
		if(this.scalar != null){
			MatrixBlock scMb = this.scalar.computeEOpNode(inputs,numThreads,LOG);
			scalarObjects.add(new DoubleObject(scMb.get(0,0)));
		}
        MatrixBlock out = r.execute(fuseInputs, scalarObjects, new MatrixBlock(), numThreads);
        if( isResultA ||  isResultB || isResultZ)
            ensureMatrixBlockColumnVector(out);
        return out;

    }

    @Override
    public void reorderChildren(Character outChar1, Character outChar2) {

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

