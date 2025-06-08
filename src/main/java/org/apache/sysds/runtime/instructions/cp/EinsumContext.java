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

import org.apache.sysds.runtime.matrix.data.MatrixBlock;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;

public class EinsumContext {
    public Integer outRows;
    public Integer outCols;
    public HashMap<Character, Integer> charToDimensionSizeInt;
    public String equationString;
    public Integer[] contractDims;
    public Integer[] summingDims;
    public HashSet<Character> summingChars;
    public HashSet<Character> contractDimsSet;

    public static EinsumContext getEinsumContext(String eqStr, ArrayList<MatrixBlock> inputs){
        EinsumContext res = new EinsumContext();

        res.equationString = eqStr;
        int i = 0;
        res.charToDimensionSizeInt  = new HashMap<Character, Integer>();
        Iterator<MatrixBlock> it = inputs.iterator();
        MatrixBlock curArr = it.next();
        int arrSizeIterator=0;
        HashSet<Character> summingChars = new HashSet<>();
        Integer[] contractDims = new Integer[inputs.size()];//0==nothing, 1 = right, 2=left, 3 = both
        Integer[] summingDims = new Integer[inputs.size()];//0/null==nothing, 1 = right, 2=left, 3 = both
        HashSet<Character> contractDimsSet = new HashSet();


        int arrIt = 0;
        for (i = 0; true; i++){
            char c = eqStr.charAt(i);
            if(c=='-'){
                i+=2;
                break;
            }
            if(c==','){

                arrIt++;
                curArr = it.next();
                arrSizeIterator = 0;
            }

            else{
                if (res.charToDimensionSizeInt.containsKey(c)){
                    // just check if dims match!
                    if(arrSizeIterator==0)
                        assert (res.charToDimensionSizeInt.get(c) == curArr.getNumRows());
                    else if(arrSizeIterator==1)
                        assert (res.charToDimensionSizeInt.get(c) == curArr.getNumColumns());

                    summingChars.add(c);

                }else{
                    if(arrSizeIterator==0)
                        res.charToDimensionSizeInt.put(c, curArr.getNumRows());
                    else if(arrSizeIterator==1)
                        res.charToDimensionSizeInt.put(c, curArr.getNumColumns());
                }
                arrSizeIterator++;
            }

            //Process char
        }
        int rem = eqStr.length() - i;
        arrSizeIterator = 0;
        if (rem ==0){
            res.outRows=1;
            res.outCols=1;

            arrIt=0;
            for (i = 0; true; i++) {
                char c = eqStr.charAt(i);
                if (c == '-') {
                    break;
                }
                if (c == ',') {
                    arrIt++;
                    arrSizeIterator = 0;
                    continue;
                }

                if(summingChars.contains(c)){

                }else{
                    contractDimsSet.add(c);
                    if(contractDims[arrIt]==null){
                        contractDims[arrIt] = arrSizeIterator +1;

                    }else {
                        contractDims[arrIt] += arrSizeIterator + 1;
                    }
                }
                arrSizeIterator++;

            }
        }else if (rem == 1){
            char c1= eqStr.charAt(i);
            res.outRows=(res.charToDimensionSizeInt.get(c1));

            res.outCols=1;
            arrIt=0;
            for (i = 0; true; i++) {
                char c = eqStr.charAt(i);
                if (c == '-') {
                    break;
                }
                if (c == ',') {
                    arrIt++;
                    arrSizeIterator = 0;
                    continue;
                }

                if(summingChars.contains(c)){

                    if(summingDims[arrIt] == null){
                        summingDims[arrIt]=arrSizeIterator +1; // it=0->add 1, it==1->add 2
                    }else{
                        summingDims[arrIt]+=arrSizeIterator +1; // it=0->add 1, it==1->add 2

                    }
                }else if(c==c1){
                    // this dim is remaining
                }else{
                    contractDimsSet.add(c);

                    if(contractDims[arrIt]==null){
                        contractDims[arrIt]=arrSizeIterator +1;

                    }else {
                        contractDims[arrIt] += arrSizeIterator + 1;
                    }

                }
                arrSizeIterator++;

            }
        }else if (rem==2){
            char c1= eqStr.charAt(i);
            char c2= eqStr.charAt(i+1);
            res.outRows=(res.charToDimensionSizeInt.get(c1));
            res.outCols=(res.charToDimensionSizeInt.get(c2));

            arrIt=0;
            for (i = 0; true; i++) {
                char c = eqStr.charAt(i);
                if (c == '-') {
                    break;
                }
                if (c == ',') {
                    arrIt++;
                    arrSizeIterator = 0;
                    continue;

                }

                if(summingChars.contains(c)){
                    if(summingDims[arrIt] == null){
                        summingDims[arrIt]=arrSizeIterator +1; // it=0->add 1, it==1->add 2
                    }else{
                        summingDims[arrIt]+=arrSizeIterator +1; // it=0->add 1, it==1->add 2

                    }
                }else if(c==c1 || c==c2){
                    // this dim is remaining
                }else{
                    contractDimsSet.add(c);

                    if(contractDims[arrIt]==null){
                        contractDims[arrIt]=arrSizeIterator +1;

                    }else {
                        contractDims[arrIt] += arrSizeIterator + 1;
                    }
                }
                arrSizeIterator++;

            }
        }else{
            throw new RuntimeException("output dim > 2 not supported for now");
        }
        res.contractDims=contractDims;
        res.contractDimsSet = contractDimsSet;
        res.summingDims=summingDims;

        res.summingChars = summingChars;
        return res;
    }
}
