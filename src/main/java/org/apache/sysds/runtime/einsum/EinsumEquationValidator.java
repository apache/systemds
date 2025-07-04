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

import org.apache.commons.lang3.tuple.Triple;
import org.apache.sysds.common.Types;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.parser.Identifier;
import org.apache.sysds.parser.LanguageException;
import org.apache.sysds.parser.ParseInfo;

import java.util.HashMap;
import java.util.Iterator;
import java.util.List;

public class EinsumEquationValidator {

    public static <HopOrIdentifier extends ParseInfo> Triple<Long, Long, Types.DataType> validateEinsumEquationAndReturnDimensions(String equationString, List<HopOrIdentifier> expressionsOrIdentifiers) throws LanguageException {
        String[] eqStringParts = equationString.split("->"); // length 2 if "...->..." , length 1 if "...->"
        boolean isResultScalar = eqStringParts.length == 1;

        if(expressionsOrIdentifiers == null)
            throw new RuntimeException("Einsum: called validateEinsumAndReturnDimensions with null list");

        HashMap<Character, Long> charToDimensionSize = new HashMap<>();
        Iterator<HopOrIdentifier> it = expressionsOrIdentifiers.iterator();
        HopOrIdentifier currArr = it.next();
        int arrSizeIterator = 0;
        int numberOfMatrices = 1;
        for (int i = 0; i < eqStringParts[0].length(); i++) {
            char c = equationString.charAt(i);
            if(c==' ') continue;
            if(c==','){
                if(!it.hasNext())
                    throw new LanguageException("Einsum: Provided less operands than specified in equation str");
                currArr = it.next();
                arrSizeIterator = 0;
                numberOfMatrices++;
            } else{
                long thisCharDimension = getThisCharDimension(currArr, arrSizeIterator);
                if (charToDimensionSize.containsKey(c)){
                    if (charToDimensionSize.get(c) != thisCharDimension)
                        throw new LanguageException("Einsum: Character '" + c + "' expected to be dim " + charToDimensionSize.get(c) + ", but found " + thisCharDimension);
                }else{
                    charToDimensionSize.put(c, thisCharDimension);
                }
                arrSizeIterator++;
            }
        }
        if (expressionsOrIdentifiers.size() - 1 > numberOfMatrices)
            throw new LanguageException("Einsum: Provided more operands than specified in equation str");

        if (isResultScalar)
            return Triple.of(-1l,-1l, Types.DataType.SCALAR);

        int numberOfOutDimensions = 0;
        Character dim1Char = null;
        long dim1 = 1;
        long dim2 = 1;
        for (int i = 0; i < eqStringParts[1].length(); i++) {
            char c = eqStringParts[1].charAt(i);
            if (c == ' ') continue;
            if (numberOfOutDimensions == 0) {
                dim1Char = c;
                dim1 = charToDimensionSize.get(c);
            } else {
                if(c==dim1Char) throw new LanguageException("Einsum: output character "+c+" provided multiple times");
                dim2 = charToDimensionSize.get(c);
            }
            numberOfOutDimensions++;
        }
        if (numberOfOutDimensions > 2) {
            throw new LanguageException("Einsum: output matrices with with no. dims > 2 not supported");
        } else {
            return Triple.of(dim1, dim2, Types.DataType.MATRIX);
        }
    }

    public static Types.DataType validateEinsumEquationNoDimensions(String equationString, int numberOfMatrixInputs) throws LanguageException {
        String[] eqStringParts = equationString.split("->"); // length 2 if "...->..." , length 1 if "...->"
        boolean isResultScalar = eqStringParts.length == 1;

        int numberOfMatrices = 1;
        for (int i = 0; i < eqStringParts[0].length(); i++) {
            char c = eqStringParts[0].charAt(i);
            if(c == ' ') continue;
            if(c == ',')
                numberOfMatrices++;
        }
        if(numberOfMatrixInputs != numberOfMatrices){
            throw  new LanguageException("Einsum: Invalid number of parameters, given: " + numberOfMatrixInputs + ", expected: " + numberOfMatrices);
        }

        if(isResultScalar){
            return Types.DataType.SCALAR;
        }else {
            int numberOfDimensions = 0;
            Character dim1Char = null;
            for (int i = 0; i < eqStringParts[1].length(); i++) {
                char c = eqStringParts[i].charAt(i);
                if(c == ' ') continue;
                numberOfDimensions++;
                if (numberOfDimensions == 1 && c == dim1Char)
                    throw new LanguageException("Einsum: output character "+c+" provided multiple times");
                dim1Char = c;
            }

            if (numberOfDimensions > 2) {
                throw new LanguageException("Einsum: output matrices with with no. dims > 2 not supported");
            } else {
                return Types.DataType.MATRIX;
            }
        }
    }

    private static <HopOrIdentifier extends ParseInfo> long getThisCharDimension(HopOrIdentifier currArr, int arrSizeIterator) {
        long thisCharDimension;
        if(currArr instanceof Hop){
            thisCharDimension = arrSizeIterator == 0 ? ((Hop) currArr).getDim1()  : ((Hop) currArr).getDim2();
        } else if(currArr instanceof Identifier){
            thisCharDimension = arrSizeIterator == 0 ? ((Identifier) currArr).getDim1()  : ((Identifier) currArr).getDim2();
        } else {
            throw new RuntimeException("validateEinsumAndReturnDimensions called with expressions that are not Hop or Identifier: "+ currArr == null ? "null" : currArr.getClass().toString());
        }
        return thisCharDimension;
    }
}
