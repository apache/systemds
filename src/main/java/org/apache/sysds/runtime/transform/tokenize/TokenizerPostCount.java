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

package org.apache.sysds.runtime.transform.tokenize;

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.wink.json4j.JSONException;
import org.apache.wink.json4j.JSONObject;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class TokenizerPostCount implements TokenizerPost{

    private static final long serialVersionUID = 6382000606237705019L;
    private final Params params;
    private final int numIdCols;
    private final int maxTokens;
    private final boolean wideFormat;

    static class Params implements Serializable {

        private static final long serialVersionUID = 5121697674346781880L;

        public boolean sort_alpha = false;

        public Params(JSONObject json) throws JSONException {
            if (json != null && json.has("sort_alpha")) {
                this.sort_alpha = json.getBoolean("sort_alpha");
            }
        }
    }

    public TokenizerPostCount(JSONObject params, int numIdCols, int maxTokens, boolean wideFormat) throws JSONException {
        this.params = new Params(params);
        this.numIdCols = numIdCols;
        this.maxTokens = maxTokens;
        this.wideFormat = wideFormat;
    }

    @Override
    public FrameBlock tokenizePost(List<Tokenizer.DocumentToTokens> tl, FrameBlock out) {
        for (Tokenizer.DocumentToTokens docToToken: tl) {
            List<Object> keys = docToToken.keys;
            List<Tokenizer.Token> tokenList = docToToken.tokens;
            // Creating the counts for BoW
            Map<String, Long> tokenCounts = tokenList.stream().collect(Collectors.groupingBy(token ->
                token.textToken, Collectors.counting()));
            // Remove duplicate strings
            Stream<String> distinctTokenStream = tokenList.stream().map(token -> token.textToken).distinct();
            if (params.sort_alpha) {
                // Sort alphabetically
                distinctTokenStream = distinctTokenStream.sorted();
            }
            List<String> outputTokens = distinctTokenStream.collect(Collectors.toList());

            int numTokens = 0;
            for (String token: outputTokens) {
                if (numTokens >= maxTokens) {
                    break;
                }
                // Create a row per token
                long count = tokenCounts.get(token);
                List<Object> rowList = new ArrayList<>(keys);
                rowList.add(token);
                rowList.add(count);
                Object[] row = new Object[rowList.size()];
                rowList.toArray(row);
                out.appendRow(row);
                numTokens++;
            }
        }

        return out;
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

    public long getNumRows(long inRows) {
        if (wideFormat) {
            return inRows;
        } else {
            return inRows * maxTokens;
        }
    }

    public long getNumCols() {
        return this.getOutSchema().length;
    }
}
