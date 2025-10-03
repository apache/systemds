package org.apache.sysds.runtime.einsum;

import org.apache.sysds.runtime.instructions.cp.EinsumCPInstruction;
import org.apache.sysds.runtime.matrix.data.LibMatrixMult;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

public class EOpNodeEinsumFuse extends EOpNode {
    public static final int AB_index=0;
    public static final int BA_index=1;
    public static final int B_index=2;
    public static final int XB_index=3;
    public static final int BX_index=4;
    public static final int A_index=5;
    public static final int XA_index=6;
    public static final int AX_index=7;
    public static final int AZ_index=8;
    public enum EinsumRewriteType{
        // B -> row*row, A -> row*scalar
        AB_BA_B_A__AB,
        AB_BA_B_A__B,
        AB_BA_B_A__A,
        AB_BA_B_A__,

        // scalar from row(AB).dot(B) multiplied by row(AZ)
        AB_BA_B_A_AZ__Z,

        // AC: last step is outer matrix multiplication using vector C
        AB_BA_B_A_AZ__BZ,
        AB_BA_B_A_AZ__ZB,

//        // outer matrix multiplication using vector C and vector Z
//        AB_BA_B_A_AZ_AC__ZC,
//        AB_BA_B_A_AZ_AC__CZ,
    }

    public final EinsumRewriteType einsumRewriteType;
    public final List<List<EOpNode>> operands;

    private EOpNodeEinsumFuse(Character c1, Character c2, EinsumRewriteType einsumRewriteType, List<EOpNode>... operands) {
        super(c1,c2);
        this.einsumRewriteType = einsumRewriteType;
        this.operands = Arrays.asList(operands);
    }

    public static EOpNodeEinsumFuse match(ArrayList<EOpNode> operands, Character outChar1, Character outChar2,/*, Set<Character> simplySummableChars,*/ ArrayList<EOpNode> ret, HashMap<Character, Integer> charToOccurences, HashMap<Character, Integer> charToSize){
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

        ArrayList<EOpNode> AXs = new ArrayList<>();
        ArrayList<EOpNode> XAs = new ArrayList<>();
        ArrayList<EOpNode> BXs = new ArrayList<>();
        ArrayList<EOpNode> XBs = new ArrayList<>();
        ArrayList<EOpNode> AZs = new ArrayList<>();
//        ArrayList<EOpNode> ACs = new ArrayList<>();
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

            AXs = new ArrayList<>();
            XAs = new ArrayList<>();
            BXs = new ArrayList<>();
            XBs = new ArrayList<>();
            AZs = new ArrayList<>();
            Character z = null;
            pass=true;
            int AZsCounter = 0;
            HashSet<String> AZCandidates = new HashSet<>();

            for (String chars : charsToMatrices.keySet()) {
                if (chars.equals(ABcandidate) || chars.equals(BA)) {
//                    ABsCounter++;
                    continue;
                }

                if(chars.length()==1){
                    if(chars.charAt(0)==a){
//                        AsCounter++;
                    }else if(chars.charAt(0)==b){
//                        BsCounter++;
                    }
                    continue;
                    //always ok
                }else{
                    if(a==chars.charAt(1) && b==chars.charAt(0)){ //BA
//                        ABsCounter++;
                        continue;
                    }
                    if(chars.charAt(0)==a){
                        //AZ
                        AZsCounter++;
                        AZCandidates.add(chars);
                    }
                    else if(chars.charAt(0)==b){
                        // BZ, todo, maybe transpose ab into ba
                        pass = false;
                        break;
                    }
                    else if(chars.charAt(1)==a){
                        //ZA, maybe its small enough that it can be tranposed? but then not impactful as the bigger A, the more sense to fuse AZ?
                            pass = false;
                            break;
                    }
                    else if(chars.charAt(1)==b){
                        // ZB
                        pass = false;
                        break;
                    }
                }
            }
            if(pass){

                AB = ABcandidate;
                String A = ""+a;
                String B = ""+b;
                int ABsCounter = charsToMatrices.get(ABcandidate).size()+(charsToMatrices.containsKey(BA) ?  charsToMatrices.get(BA).size() : 0);
//                int AZsCounter = AZs.size();
                int AsCounter = (charsToMatrices.containsKey(A) ?  charsToMatrices.get(A).size() : 0);
                int BsCounter = (charsToMatrices.containsKey(B) ?  charsToMatrices.get(B).size() : 0);
                if(AsCounter==0 && BsCounter==0 && ABsCounter<2){
                    pass=false;
                    continue;
                }
                int usedBsCount = BsCounter+ABsCounter;
                doSumB = charToOccurences.get(b)==usedBsCount && (outChar1 == null || b!=outChar1) && (outChar2 == null || b!=outChar2);

                if(AZCandidates.size()==1){
//                    if(!doSumB){
//                        pass=false;
//                        continue;
//                    }
                    int usedAsCount = AsCounter+ABsCounter+AZsCounter;
                    doSumA = charToOccurences.get(a)==usedAsCount && (outChar1 == null || a!=outChar1) && (outChar2 == null || a!=outChar2);
                    if(!doSumA){ // cant do AZ
                        break;// just do AB,B,A ->AB / A
                    }else {
                        AZs = charsToMatrices.get(AZCandidates.iterator().next());
                        break;//ok
                    }
                } else if (AZCandidates.size()>=2) {
                    doSumA = false;
                    if(doSumB){
                        pass=true;
                        break; // can do it, it will create AB,B,A -> A, that will be consumed by some AZ later
                    }
                    pass=false;
                    continue;

                }
                int usedAsCount = AsCounter+ABsCounter;
                doSumA = charToOccurences.get(a)==usedAsCount && (outChar1 == null || a!=outChar1) && (outChar2 == null || a!=outChar2);

                break;
            }
        }

        if(!pass){
            return null;
        }
        String B = AB.substring(1,2);
        String A = AB.substring(0,1);
        char a = A.charAt(0);
        char b = B.charAt(0);
        Character c1 = null;
        Character c2 = null;
        EinsumRewriteType t = null;

        if(!AZs.isEmpty()){
//            Character azC1 = AZs.get(0).c1;
            Character azC2 = AZs.get(0).c2;
//            c1 = AZs.get(0).c2;
            if(doSumB) {
                t = EinsumRewriteType.AB_BA_B_A_AZ__Z;
                c1 = azC2;

            }
            else if (EinsumCPInstruction.FUSE_OUTER_MULTIPLY) {
                if(LibMatrixMult.isSkinnyRightHandSide(charToSize.get(AB.charAt(0)), charToSize.get(AB.charAt(1)), charToSize.get(azC2), charToSize.get(AB.charAt(1)),false)||
                        LibMatrixMult.isSkinnyRightHandSide(charToSize.get(AB.charAt(0)), azC2, charToSize.get(AB.charAt(1)), charToSize.get(azC2),false)) {
                    // ideally this can be changed later by parent,depending on need
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

                }
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

        ArrayList<EOpNode> ABs=charsToMatrices.containsKey(AB) ? charsToMatrices.get(AB) : new ArrayList<>();
        ArrayList<EOpNode> BAs=charsToMatrices.containsKey(BA) ? charsToMatrices.get(BA) : new ArrayList<>();
        ArrayList<EOpNode> Bs=charsToMatrices.containsKey(B) ? charsToMatrices.get(B) : new ArrayList<>();
        ArrayList<EOpNode> As=charsToMatrices.containsKey(A) ? charsToMatrices.get(A) : new ArrayList<>();

        usedOperands.addAll(ABs);
        usedOperands.addAll(BAs);
        usedOperands.addAll(Bs);
        usedOperands.addAll(As);
        usedOperands.addAll(XBs);
        usedOperands.addAll(BXs);
        usedOperands.addAll(XAs);
        usedOperands.addAll(AXs);
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

        var e = new EOpNodeEinsumFuse(c1, c2, t,
                ABs,
                BAs,
                Bs,
                XBs,
                BXs,
                As,
                XAs,
                AXs,
                AZs,
                Zs
        );
        ret.add(e);
        return e;
    }
}

