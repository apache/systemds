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

import org.apache.sysds.runtime.matrix.data.MatrixBlock;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;


public class EinsumContext {
    public enum ContractDimensions {
        CONTRACT_LEFT,
        CONTRACT_RIGHT,
        CONTRACT_BOTH,
    }
    public Integer outRows;
    public Integer outCols;
    public Character outChar1;
    public Character outChar2;
    public HashMap<Character, Integer> charToDimensionSize;
    public String equationString;
    public boolean[] diagonalInputs;
    public HashSet<Character> summingChars;
    public HashSet<Character> contractDimsSet;
    public ContractDimensions[] contractDims;
    public ArrayList<String> newEquationStringInputsSplit;
    public HashMap<Character, ArrayList<Integer>> characterAppearanceIndexes; // for each character, this tells in which inputs it appears

    private EinsumContext(){};
    public static EinsumContext getEinsumContext(String eqStr, ArrayList<MatrixBlock> inputs){
        EinsumContext res = new EinsumContext();

        res.equationString = eqStr;
        res.charToDimensionSize = new HashMap<Character, Integer>();
        HashSet<Character> summingChars = new HashSet<>();
        ContractDimensions[] contractDims = new ContractDimensions[inputs.size()];
        boolean[] diagonalInputs = new boolean[inputs.size()]; // all false by default
        HashSet<Character> contractDimsSet = new HashSet<>();
        HashMap<Character, ArrayList<Integer>> partsCharactersToIndices = new HashMap<>();
        ArrayList<String> newEquationStringSplit = new ArrayList<>();

        Iterator<MatrixBlock> it = inputs.iterator();
        MatrixBlock curArr = it.next();
        int arrSizeIterator = 0;
        int arrayIterator = 0;
        int i;
        // first iteration through string: collect information on character-size and what characters are summing characters
        for (i = 0; true; i++) {
            char c = eqStr.charAt(i);
            if(c == '-'){
                i+=2;
                break;
            }
            if(c == ','){
                arrayIterator++;
                curArr = it.next();
                arrSizeIterator = 0;
            }
            else{
                if (res.charToDimensionSize.containsKey(c)) { // sanity check if dims match, this is already checked at validation
                    if(arrSizeIterator == 0 && res.charToDimensionSize.get(c) != curArr.getNumRows())
                        throw new RuntimeException("Einsum: character "+c+" has multiple conflicting sizes");
                    else if(arrSizeIterator == 1 && res.charToDimensionSize.get(c) != curArr.getNumColumns())
                        throw new RuntimeException("Einsum: character "+c+" has multiple conflicting sizes");
                    summingChars.add(c);
                } else {
                    if(arrSizeIterator == 0)
                        res.charToDimensionSize.put(c, curArr.getNumRows());
                    else if(arrSizeIterator == 1)
                        res.charToDimensionSize.put(c, curArr.getNumColumns());
                }

                arrSizeIterator++;
            }
        }

        int numOfRemainingChars = eqStr.length() - i;

        if (numOfRemainingChars > 2)
            throw new RuntimeException("Einsum: dim > 2 not supported");

        arrSizeIterator = 0;

        Character outChar1 = numOfRemainingChars > 0 ? eqStr.charAt(i) : null;
        Character outChar2 = numOfRemainingChars > 1 ? eqStr.charAt(i+1) : null;
        res.outRows=(numOfRemainingChars > 0 ? res.charToDimensionSize.get(outChar1) : 1);
        res.outCols=(numOfRemainingChars > 1 ? res.charToDimensionSize.get(outChar2) : 1);

        arrayIterator=0;
        // second iteration through string: collect remaining information
        for (i = 0; true; i++) {
            char c = eqStr.charAt(i);
            if (c == '-') {
                break;
            }
            if (c == ',') {
                arrayIterator++;
                arrSizeIterator = 0;
                continue;
            }
            String s = "";

            if(summingChars.contains(c)) {
                s+=c;
                if(!partsCharactersToIndices.containsKey(c))
                    partsCharactersToIndices.put(c, new ArrayList<>());
                partsCharactersToIndices.get(c).add(arrayIterator);
            }
            else if((outChar1 != null && c == outChar1) || (outChar2 != null && c == outChar2)) {
                s+=c;
            }
            else {
                contractDimsSet.add(c);
                contractDims[arrayIterator] = ContractDimensions.CONTRACT_LEFT;
            }

            if(i + 1 < eqStr.length()) { // process next character together
                char c2 = eqStr.charAt(i + 1);
                i++;
                if (c2 == '-') { newEquationStringSplit.add(s); break;}
                if (c2 == ',') { arrayIterator++; newEquationStringSplit.add(s); continue; }

                if (c2 == c){
                    diagonalInputs[arrayIterator] = true;
                    if (contractDims[arrayIterator] == ContractDimensions.CONTRACT_LEFT) contractDims[arrayIterator] = ContractDimensions.CONTRACT_BOTH;
                }
                else{
                    if(summingChars.contains(c2)) {
                        s+=c2;
                        if(!partsCharactersToIndices.containsKey(c2))
                            partsCharactersToIndices.put(c2, new ArrayList<>());
                        partsCharactersToIndices.get(c2).add(arrayIterator);
                    }
                    else if((outChar1 != null && c2 == outChar1) || (outChar2 != null && c2 == outChar2)) {
                        s+=c2;
                    }
                    else {
                        contractDimsSet.add(c2);
                        contractDims[arrayIterator] = contractDims[arrayIterator] == ContractDimensions.CONTRACT_LEFT ? ContractDimensions.CONTRACT_BOTH : ContractDimensions.CONTRACT_RIGHT;
                    }
                }
            }
            newEquationStringSplit.add(s);
            arrSizeIterator++;
        }

        res.contractDims = contractDims;
        res.contractDimsSet = contractDimsSet;
        res.diagonalInputs = diagonalInputs;
        res.summingChars = summingChars;
        res.outChar1 = outChar1;
        res.outChar2 = outChar2;
        res.newEquationStringInputsSplit = newEquationStringSplit;
        res.characterAppearanceIndexes = partsCharactersToIndices;
        return res;
    }
}
