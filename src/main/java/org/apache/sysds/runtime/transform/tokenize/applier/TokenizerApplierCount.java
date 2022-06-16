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

package org.apache.sysds.runtime.transform.tokenize.applier;

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.transform.tokenize.DocumentRepresentation;
import org.apache.sysds.runtime.transform.tokenize.Token;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.wink.json4j.JSONException;
import org.apache.wink.json4j.JSONObject;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.apache.sysds.runtime.util.UtilFunctions.getEndIndex;

public class TokenizerApplierCount extends TokenizerApplier {

    private static final long serialVersionUID = 6382000606237705019L;
    public boolean sort_alpha = false;

    private List<Map<String, Integer>> counts;

    public TokenizerApplierCount(int numIdCols, int maxTokens, boolean wideFormat, boolean applyPadding, JSONObject params) throws JSONException {
        super(numIdCols, maxTokens, wideFormat, applyPadding);
        if (params != null && params.has("sort_alpha")) {
            this.sort_alpha = params.getBoolean("sort_alpha");
        }
    }

    @Override
    public int getNumRows(DocumentRepresentation[] internalRepresentation) {
        if(wideFormat)
            return internalRepresentation.length;
        if(applyPadding)
            return maxTokens * internalRepresentation.length;
        return counts.stream().mapToInt(hashMap -> Math.min(hashMap.size(), maxTokens)).sum();
    }

    @Override
    public void allocateInternalMeta(int numDocuments) {
        counts = new ArrayList<>(Collections.nCopies(numDocuments,null));
    }

    @Override
    public void build(DocumentRepresentation[] internalRepresentation, int inputRowStart, int blk){
        int endIndex = getEndIndex(internalRepresentation.length, inputRowStart, blk);
        for(int i = inputRowStart; i < endIndex; i++){
            Map<String, Integer> tokenCounts = new HashMap<>();
            for(Token token: internalRepresentation[i].tokens){
                String txt = token.toString();
                Integer count = tokenCounts.getOrDefault(txt, null);
                if(count != null)
                    tokenCounts.put(txt, count + 1);
                else
                    tokenCounts.put(txt, 1);
            }
            counts.set(i, tokenCounts);
        }
    }

    @Override
    public int applyInternalRepresentation(DocumentRepresentation[] internalRepresentation, FrameBlock out, int inputRowStart, int blk) {
        int endIndex = getEndIndex(internalRepresentation.length, inputRowStart, blk);
        int outputRow = getOutputRow(inputRowStart, counts);
        for(int i = inputRowStart; i < endIndex; i++) {
            List<Object> keys = internalRepresentation[i].keys;
            // Creating the counts for BoW
            Map<String, Integer> tokenCounts = counts.get(i);
            // Remove duplicate strings
            Collection<String> distinctTokens = tokenCounts.keySet();
            if (this.sort_alpha) {
                // Sort alphabetically
                distinctTokens = new TreeSet<>(distinctTokens);
            }

            int numTokens = 0;
            for (String token: distinctTokens) {
                if (numTokens >= maxTokens) {
                    break;
                }
                int col = setKeys(outputRow, keys, out);
                // Create a row per token
                long count = tokenCounts.get(token);
                out.set(outputRow, col, token);
                out.set(outputRow, col+1, count);
                outputRow++;
                numTokens++;
            }
            if(applyPadding){
               outputRow = applyPaddingLong(outputRow, numTokens, keys, out, PADDING_STRING, -1);
            }
        }
        return outputRow;
    }

    @Override
    public Types.ValueType[] getOutSchema() {
        if (wideFormat) {
            throw new IllegalArgumentException("Wide Format is not supported for Count Representation.");
        }
        // Long format only depends on numIdCols
        Types.ValueType[]  schema = UtilFunctions.nCopies(numIdCols + 2,Types.ValueType.STRING );
        schema[numIdCols + 1] = Types.ValueType.INT64;
        return schema;
    }

}
