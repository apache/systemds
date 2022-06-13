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

package org.apache.sysds.runtime.transform.tokenize.builder;

import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.transform.tokenize.Tokenizer;
import org.apache.sysds.runtime.util.DependencyTask;
import org.apache.wink.json4j.JSONException;
import org.apache.wink.json4j.JSONObject;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public class TokenizerBuilderNgram implements TokenizerBuilder {

    private static final long serialVersionUID = -6297904316677723802L;

    private final TokenizerBuilderWhitespaceSplit tokenizerPreWhitespaceSplit;
    private final Params params;

    static class Params implements Serializable {

        private static final long serialVersionUID = -6516419749810062677L;

        public int minGram = 1;
        public int maxGram = 2;

        public Params(JSONObject json) throws JSONException {
            if (json != null && json.has("min_gram")) {
                this.minGram = json.getInt("min_gram");
            }
            if (json != null && json.has("max_gram")) {
                this.maxGram = json.getInt("max_gram");
            }
        }
    }

    public TokenizerBuilderNgram(List<Integer> idCols, int tokenizeCol, JSONObject params) throws JSONException {
        this.tokenizerPreWhitespaceSplit = new TokenizerBuilderWhitespaceSplit(idCols, tokenizeCol, params);
        this.params = new Params(params);
    }

    public List<Tokenizer.Token> wordTokenToNgrams(Tokenizer.Token wordTokens) {
        List<Tokenizer.Token> ngramTokens = new ArrayList<>();

        int tokenLen = wordTokens.textToken.length();
        int startPos = params.minGram - params.maxGram;
        int endPos = Math.max(tokenLen - params.minGram, startPos);

        for (int i = startPos; i <= endPos; i++) {
            int startSlice = Math.max(i, 0);
            int endSlice = Math.min(i + params.maxGram, tokenLen);
            String substring = wordTokens.textToken.substring(startSlice, endSlice);
            long tokenStart = wordTokens.startIndex + startSlice;
            ngramTokens.add(new Tokenizer.Token(substring, tokenStart));
        }

        return ngramTokens;
    }

    public List<Tokenizer.Token> wordTokenListToNgrams(List<Tokenizer.Token> wordTokens) {
        List<Tokenizer.Token> ngramTokens = new ArrayList<>();

        for (Tokenizer.Token wordToken: wordTokens) {
            List<Tokenizer.Token> ngramTokensForWord = wordTokenToNgrams(wordToken);
            ngramTokens.addAll(ngramTokensForWord);
        }
        return ngramTokens;
    }

    @Override
    public void createInternalRepresentation(FrameBlock in, List<Tokenizer.DocumentRepresentation> internalRepresentation) {
        tokenizerPreWhitespaceSplit.createInternalRepresentation(in, internalRepresentation);

        List<Tokenizer.DocumentRepresentation> docToNgramTokens = new ArrayList<>();
        for (Tokenizer.DocumentRepresentation docToTokens: internalRepresentation) {
            List<Object> keys = docToTokens.keys;
            List<Tokenizer.Token> wordTokens = docToTokens.tokens;
            List<Tokenizer.Token> ngramTokens = wordTokenListToNgrams(wordTokens);
            docToNgramTokens.add(new Tokenizer.DocumentRepresentation(keys, ngramTokens));
        }
    }

    @Override
    public List<DependencyTask<?>> getTasks(FrameBlock in, List<Tokenizer.DocumentRepresentation> internalRepresentation, int k) {
        return null;
    }
}
