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
import org.apache.sysds.runtime.transform.tokenize.DocumentRepresentation;
import org.apache.sysds.runtime.transform.tokenize.Token;
import org.apache.sysds.runtime.transform.tokenize.Tokenizer;
import org.apache.sysds.runtime.util.DependencyTask;
import org.apache.wink.json4j.JSONException;
import org.apache.wink.json4j.JSONObject;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.Callable;
import java.util.stream.Collectors;

import static org.apache.sysds.runtime.util.UtilFunctions.getBlockSizes;
import static org.apache.sysds.runtime.util.UtilFunctions.getEndIndex;

public class TokenizerBuilderWhitespaceSplit extends TokenizerBuilder {

    private static final long serialVersionUID = 539127244034913364L;

    private final int[] idCols;
    private final int tokenizeCol;

    public String regex = "\\s+"; // whitespace

    public TokenizerBuilderWhitespaceSplit(int[] idCols, int tokenizeCol, JSONObject params) throws JSONException {
        if (params != null && params.has("regex")) {
            this.regex = params.getString("regex");
        }
        this.idCols = idCols;
        this.tokenizeCol = tokenizeCol;
    }

    public List<Token> splitToTokens(String text) {
        List<Token> tokenList = new ArrayList<>();
        if(text == null)
            return tokenList;
        String[] textTokens = text.split(this.regex);
        int curIndex = 0;
        for(String textToken: textTokens) {
            if(Objects.equals(textToken, "")){
                continue;
            }
            int tokenIndex = text.indexOf(textToken, curIndex);
            curIndex = tokenIndex;
            tokenList.add(new Token(textToken, tokenIndex));
        }
        return tokenList;
    }

    @Override
    public void createInternalRepresentation(FrameBlock in, DocumentRepresentation[] internalRepresentation, int rowStart, int blk) {
        int endIndex = getEndIndex(in.getNumRows(), rowStart, blk);
        for (int i = rowStart; i < endIndex; i++) {
            String text = in.getString(i, tokenizeCol - 1);
            List<Token> tokenList = splitToTokens(text);
            List<Object> keys = new ArrayList<>();
            for (Integer idCol : idCols) {
                Object key = in.get(i, idCol - 1);
                keys.add(key);
                internalRepresentation[i] = new DocumentRepresentation(keys, tokenList);
            }
        }
    }
}
