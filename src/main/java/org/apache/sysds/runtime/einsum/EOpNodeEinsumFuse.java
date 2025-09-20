package org.apache.sysds.runtime.einsum;

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
        // inputops__output              'X' = simplySumDim
        AB_BA_B_XB_BX_A_XA_AX__AB,
        AB_BA_B_XB_BX_A_XA_AX__B,
        AB_BA_B_XB_BX_A_XA_AX__A,
        AB_BA_B_XB_BX_A_XA_AX__,

        AB_BA_B_XB_BX_A_XA_AX_AZ__Z
    }
    public enum EinsumRewriteType_v2{ // option 2 without X dims
        AB_BA_B_A__AB,
        AB_BA_B_A__B,
        AB_BA_B_A__A,

        AB_BA_B_A_AZ__Z
    }
    public final EinsumRewriteType einsumRewriteType;
    public final List<List<EOpNode>> operands;

    private EOpNodeEinsumFuse(Character c1, Character c2, EinsumRewriteType einsumRewriteType, List<EOpNode>... operands) {
        super(c1,c2);
        this.einsumRewriteType = einsumRewriteType;
        this.operands = Arrays.asList(operands);
    }

    public static EOpNodeEinsumFuse match(ArrayList<EOpNode> operands, Character outChar1, Character outChar2,/*, Set<Character> simplySummableChars,*/ ArrayList<EOpNode> ret, HashMap<Character, Integer> charToOccurences){
        //precompute
        HashSet<String> matricesChars = new HashSet<>();
        HashMap<String, ArrayList<EOpNode>> charsToMatrices = new HashMap<>();
        HashMap<Character, Integer> charsToNumberOfOperands = new HashMap<>();

        for (EOpNode operand1 : operands) {
            String k;
//todo remove and use input map charToOccurences
            if (charsToNumberOfOperands.containsKey(operand1.c1)) {
                charsToNumberOfOperands.put(operand1.c1, charsToNumberOfOperands.get(operand1.c1) + 1);
            } else {
                charsToNumberOfOperands.put(operand1.c1, 1);
            }

            if (operand1.c2 != null) {
                k = operand1.c1.toString() + operand1.c2;
                matricesChars.add(k);
                if (charsToNumberOfOperands.containsKey(operand1.c2)) {
                    charsToNumberOfOperands.put(operand1.c2, charsToNumberOfOperands.get(operand1.c2) + 1);
                } else {
                    charsToNumberOfOperands.put(operand1.c2, 1);
                }
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

            pass=true;


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
                    if(a==chars.charAt(1) && b==chars.charAt(0)){
//                        ABsCounter++;
                        //BA
                        continue;
                    }
                    if(chars.charAt(0)==a){
                        if(charsToNumberOfOperands.get(chars.charAt(1))==1){
                            if(chars.charAt(1)!= outChar1 && chars.charAt(1) != outChar2) {
                                AXs.addAll(charsToMatrices.get(chars));
//                                AsCounter++;
                                continue;
                            }else{
                                if(AZs.size()==0){
                                    AZs = charsToMatrices.get(chars);
                                    continue;
                                }
                                pass = false;
                                break;
                            }
                        }else{
                            //dont allow for now, in theory AZ,Z or AZ,AZ would also work, but for now do them separately
                            pass = false;
                            break;
                        }
                    }
                    else if(chars.charAt(0)==b){
                        if(charsToNumberOfOperands.get(chars.charAt(1))==1){
                            if(chars.charAt(1)!= outChar1 && chars.charAt(1) != outChar2) {
                                BXs.addAll(charsToMatrices.get(chars));
//                                BsCounter++;
                                continue;
                            }else{
                                pass = false; // no BZ, maybe experiment later
                                break;
                            }
                        }else{
                            pass = false;
                            break;
                        }
                    }
                    else if(chars.charAt(1)==a){
                        if(charsToNumberOfOperands.get(chars.charAt(0))==1){
                            if(chars.charAt(0)!= outChar1 && chars.charAt(0) != outChar2) {
                                XAs.addAll(charsToMatrices.get(chars));
//                                AsCounter++;
                                continue;
                            }else{
                                pass = false;
                                break;
                            }
                        }else{
                            pass = false;
                            break;
                        }
                    }
                    else if(chars.charAt(1)==b){
                        if(charsToNumberOfOperands.get(chars.charAt(0))==1){
                            if(chars.charAt(0)!= outChar1 && chars.charAt(0) != outChar2) {
                                XBs.addAll(charsToMatrices.get(chars));
//                                BsCounter++;
                                continue;
                            }else{
                                pass = false;
                                break;
                            }
                        }else{
                            pass = false;
                            break;
                        }
                    }
                }
            }
            if(pass){
                AB = ABcandidate;
                String A = ""+a;
                String B = ""+b;
                int ABsCounter = charsToMatrices.get(ABcandidate).size()+(charsToMatrices.containsKey(BA) ?  charsToMatrices.get(BA).size() : 0);
                int AsCounter = (charsToMatrices.containsKey(A) ?  charsToMatrices.get(A).size() : 0) +AXs.size()+XAs.size();
                int BsCounter = (charsToMatrices.containsKey(B) ?  charsToMatrices.get(B).size() : 0)+BXs.size()+XBs.size();
                if(AsCounter==0 && BsCounter==0 && ABsCounter<2){
                    pass=false;
                    continue;
                }
                int usedAsCount = AsCounter+ABsCounter;
                int usedBsCount = BsCounter+ABsCounter;
                doSumA = charToOccurences.get(a)==usedAsCount && (outChar1 == null || a!=outChar1) && (outChar2 == null || a!=outChar2);
                doSumB = charToOccurences.get(b)==usedBsCount && (outChar1 == null || b!=outChar1) && (outChar2 == null || b!=outChar2);
                if(AZs.size()!=0) { // invalidate AZ fusion
                    if (outChar1 != null) {
                        if (a == outChar1 || b == outChar1) {
                            pass=false;
                            continue;
                        }
                    }
                    if (outChar2 != null) {
                        if (a == outChar2 || b == outChar2) {
                            pass=false;
                            continue;
                        }
                    }
                    if(!doSumA ||  !doSumB){
                        pass=false;
                        continue;
                    }
                }
                break;
            }
        }

        if(!pass){
            return null;
        }
        String B = AB.substring(1,2);
        String A = AB.substring(0,1);
        Character c1 = null;
        Character c2 = null;
        EinsumRewriteType t;

        if(AZs.size()!=0){
            c1=AZs.get(0).c2;
            t=EinsumRewriteType.AB_BA_B_XB_BX_A_XA_AX_AZ__Z;
        }
        else if(doSumA){
            if(doSumB) {
                t = EinsumRewriteType.AB_BA_B_XB_BX_A_XA_AX__;
            }
            else {
                t = EinsumRewriteType.AB_BA_B_XB_BX_A_XA_AX__B;
                c1 = AB.charAt(1);
            }
        }
        else if(doSumB){
            t= EinsumRewriteType.AB_BA_B_XB_BX_A_XA_AX__A;
            c1= AB.charAt(0);
        }
        else {
            t = EinsumRewriteType.AB_BA_B_XB_BX_A_XA_AX__AB;
            c1 = AB.charAt(0);
            c2 = AB.charAt(1);
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
                AZs
        );
        ret.add(e);
        return e;
    }
}

