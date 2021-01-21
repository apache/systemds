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

import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

public class TokenizerWhitespaceBOW extends Tokenizer {

    private static final long serialVersionUID = 9130577081982055688L;

    private final String splitRegex = "\\s+";

    protected TokenizerWhitespaceBOW(Types.ValueType[] schema, int[] colList) {
        super(schema, colList);
    }

    @Override
    public FrameBlock tokenize(FrameBlock in, FrameBlock out) {
        // First comment to internal representation
        DocumentsToTokenList documentsToTokenList = tokenizePre(in);
        // Then convert to output representation
        return tokenizePost(documentsToTokenList, out);
    }

    class Token {
        String token;
//        int start;
//        int end;

        public Token(String token) {
            this.token = token;
        }
    }

    class DocumentsToTokenList extends HashMap<String, List<Token>> {}

    public DocumentsToTokenList tokenizePre(FrameBlock in) {
        DocumentsToTokenList documentsToTokenList = new DocumentsToTokenList();

        Iterator<String[]> iterator = in.getStringRowIterator();
        iterator.forEachRemaining(s -> {
            String key = s[0];
            String text = s[1];
            String[] tokens = text.split(splitRegex);
            // Transform to Bag format internally
            List<Token> tokenList = Arrays.stream(tokens).map(Token::new).collect(Collectors.toList());
            documentsToTokenList.put(key, tokenList);
        });

        return documentsToTokenList;
    }

    public FrameBlock tokenizePost(DocumentsToTokenList tl, FrameBlock out) {
        tl.forEach((key, tokenList) -> {
            // Creating the counts for BoW
            Map<String, Long> tokenCounts = tokenList.stream().collect(Collectors.groupingBy(token -> token.token, Collectors.counting()));
            // Sort alphabetically and remove duplicates
            List<String> sortedTokens = tokenList.stream().map(token -> token.token).distinct().sorted().collect(Collectors.toList());

            for (String token: sortedTokens) {
                String count = String.valueOf(tokenCounts.get(token));
                String[] row = {key, token, count};
                out.appendRow(row);
            }
        });

        return out;
    }
}
