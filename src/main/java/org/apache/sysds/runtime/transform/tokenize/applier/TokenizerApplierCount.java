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

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.apache.sysds.runtime.util.UtilFunctions.getEndIndex;

public class TokenizerApplierCount extends TokenizerApplier {

    private static final long serialVersionUID = 6382000606237705019L;
    public boolean sort_alpha = false;


    public TokenizerApplierCount(int numIdCols, int maxTokens, boolean wideFormat, boolean applyPadding, JSONObject params) throws JSONException {
        super(numIdCols, maxTokens, wideFormat, applyPadding);
        if (params != null && params.has("sort_alpha")) {
            this.sort_alpha = params.getBoolean("sort_alpha");
        }
    }

    @Override
    public int applyInternalRepresentation(DocumentRepresentation[] internalRepresentation, FrameBlock out, int inputRowStart, int blk) {
        int endIndex = getEndIndex(internalRepresentation.length, inputRowStart, blk);
        int outputRow = Arrays.stream(internalRepresentation).limit(inputRowStart).mapToInt(doc -> applyPadding? maxTokens: doc.tokens.size()).sum();
        for(int i = inputRowStart; i < endIndex; i++) {
            List<Object> keys = internalRepresentation[i].keys;
            List<Token> tokenList = internalRepresentation[i].tokens;
            // Creating the counts for BoW
            Map<String, Long> tokenCounts = tokenList.stream().collect(Collectors.groupingBy(Token::toString, Collectors.counting()));
            // Remove duplicate strings
            Stream<String> distinctTokenStream = tokenList.stream().map(Token::toString).distinct();
            if (this.sort_alpha) {
                // Sort alphabetically
                distinctTokenStream = distinctTokenStream.sorted();
            }
            List<String> outputTokens = distinctTokenStream.collect(Collectors.toList());

            int numTokens = 0;
            for (String token: outputTokens) {
                if (numTokens >= maxTokens) {
                    break;
                }
                int col = 0;
                for(; col < keys.size(); col++){
                    out.set(outputRow, col, keys.get(col));
                }
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
