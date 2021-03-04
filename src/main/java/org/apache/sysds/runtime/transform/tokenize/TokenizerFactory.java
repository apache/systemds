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

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.wink.json4j.JSONObject;
import org.apache.wink.json4j.JSONArray;

import java.util.ArrayList;
import java.util.List;

public class TokenizerFactory {

    public static Tokenizer createTokenizer(String spec, int maxTokens) {
        Tokenizer tokenizer = null;

        try {
            //parse transform specification
            JSONObject jSpec = new JSONObject(spec);

            // tokenization needs an algorithm (with algorithm specific params)
            String algo = jSpec.getString("algo");
            JSONObject algoParams = null;
            if (jSpec.has("algo_params")) {
                algoParams = jSpec.getJSONObject("algo_params");
            }

            // tokenization needs an output representation (with representation specific params)
            String out = jSpec.getString("out");
            JSONObject outParams = null;
            if (jSpec.has("out_params")) {
                outParams = jSpec.getJSONObject("out_params");
            }

            // tokenization needs a text column to tokenize
            int tokenizeCol = jSpec.getInt("tokenize_col");

            // tokenization needs one or more idCols that define the document and are replicated per token
            List<Integer> idCols = new ArrayList<>();
            JSONArray idColsJsonArray = jSpec.getJSONArray("id_cols");
            for (int i=0; i < idColsJsonArray.length(); i++) {
                idCols.add(idColsJsonArray.getInt(i));
            }
            // Output schema is derived from specified id cols
            int numIdCols = idCols.size();

            // get difference between long and wide format
            boolean wideFormat = false;  // long format is default
            if (jSpec.has("format_wide")) {
                wideFormat = jSpec.getBoolean("format_wide");
            }

            TokenizerPre tokenizerPre;
            TokenizerPost tokenizerPost;

            // Note that internal representation should be independent from output representation

            // Algorithm to transform tokens into internal token representation
            switch (algo) {
                case "split":
                    tokenizerPre = new TokenizerPreWhitespaceSplit(idCols, tokenizeCol, algoParams);
                    break;
                case "ngram":
                    tokenizerPre = new TokenizerPreNgram(idCols, tokenizeCol, algoParams);
                    break;
                default:
                    throw new IllegalArgumentException("Algorithm {algo=" + algo + "} is not supported.");
            }

            // Transform tokens to output representation
            switch (out) {
                case "count":
                    tokenizerPost = new TokenizerPostCount(outParams, numIdCols, maxTokens, wideFormat);
                    break;
                case "position":
                    tokenizerPost = new TokenizerPostPosition(outParams, numIdCols, maxTokens, wideFormat);
                    break;
                case "hash":
                    tokenizerPost = new TokenizerPostHash(outParams, numIdCols, maxTokens, wideFormat);
                    break;
                default:
                    throw new IllegalArgumentException("Output representation {out=" + out + "} is not supported.");
            }

            tokenizer = new Tokenizer(tokenizerPre, tokenizerPost);
        }
        catch(Exception ex) {
            throw new DMLRuntimeException(ex);
        }
        return tokenizer;
    }
}
