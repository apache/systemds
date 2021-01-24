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
import org.apache.wink.json4j.JSONException;
import org.apache.wink.json4j.JSONObject;

import java.io.Serializable;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class TokenizerPostCount implements TokenizerPost{

    private static final long serialVersionUID = 6382000606237705019L;
    public Params params;

    static class Params implements Serializable {

        private static final long serialVersionUID = 5121697674346781880L;

        public boolean sort_alpha;

        public Params(JSONObject json) throws JSONException {
            if (json.has("sort_alpha")) {
                this.sort_alpha = json.getBoolean("sort_alpha");
            } else {
                this.sort_alpha = false;
            }
        }
    }

    public TokenizerPostCount(JSONObject params) throws JSONException {
        this.params = new Params(params);
    }

    @Override
    public FrameBlock tokenizePost(HashMap<String, List<Tokenizer.Token>> tl, FrameBlock out) {
        tl.forEach((key, tokenList) -> {
            // Creating the counts for BoW
            Map<String, Long> tokenCounts = tokenList.stream().collect(Collectors.groupingBy(token -> token.textToken, Collectors.counting()));
            // Remove duplicate strings
            Stream<String> distinctTokenStream = tokenList.stream().map(token -> token.textToken).distinct();
            if (params.sort_alpha) {
                // Sort alphabetically
                distinctTokenStream = distinctTokenStream.sorted();
            }
            List<String> outputTokens = distinctTokenStream.collect(Collectors.toList());

            for (String token: outputTokens) {
                long count = tokenCounts.get(token);
                Object[] row = {key, token, count};
                out.appendRow(row);
            }
        });

        return out;
    }

    @Override
    public Types.ValueType[] getOutSchema() {
        // Not sure why INT64 is required here, but CP Instruction fails otherwise
        return new Types.ValueType[]{Types.ValueType.STRING, Types.ValueType.STRING, Types.ValueType.INT64};
    }
}
