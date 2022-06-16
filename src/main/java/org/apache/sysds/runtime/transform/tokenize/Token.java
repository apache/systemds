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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Token {

    public static final String EMPTY_TOKEN = "";

    public static class SubToken {
        private final String text;
        private final long startIndex;
        private final long endIndex;

        public SubToken(String token, long startIndex) {
            this.text = token;
            this.startIndex = startIndex;
            this.endIndex = startIndex + token.length();
        }

        @Override
        public String toString() {
            return "SubToken{" +
                    "textToken='" + text + '\'' +
                    ", startIndex=" + startIndex +
                    ", endIndex=" + endIndex +
                    '}';
        }
    }

    private List<SubToken> subTokens;

    private Token(int subListSize){
        subTokens = new ArrayList<>(subListSize);
    }

    public Token(String token, long startIndex) {
        this(1);
        subTokens.add(new SubToken(token, startIndex));
    }

    public Token(List<String> tokens, List<Long> startIndex){
        this(tokens.size());
        if(tokens.size() != startIndex.size())
            throw new DMLRuntimeException("Cannot create token from mismatched input sizes");
        for(int i = 0; i < tokens.size(); i++){
            subTokens.add(new SubToken(tokens.get(i), startIndex.get(i)));
        }
    }

    public Token(List<Token> subList) {
        this(getNumSubTokens(subList));
        for(Token token: subList){
            subTokens.addAll(token.subTokens);
        }
    }

    private static int getNumSubTokens(List<Token> tokens){
        int sum = 0;
        for (Token token : tokens) {
            sum += token.getNumSubTokens();
        }
        return sum;
    }

    public int getNumSubTokens(){
        return subTokens.size();
    }

    public long getStartIndex(int subTokenIndex){
        return subTokens.get(subTokenIndex).startIndex;
    }

    @Override
    public int hashCode() {
        return toString().hashCode();
    }

    @Override
    public String toString() {
        if(subTokens.size() == 0){
            return EMPTY_TOKEN;
        }
        if(subTokens.size() == 1){
            return subTokens.get(0).text;
        }
        StringBuilder sb = new StringBuilder().append("\"('");
        for(int i = 0; i < subTokens.size(); i++){
            sb.append(subTokens.get(i).text);
            if(i < subTokens.size()-1)
                sb.append("', '");
        }
        sb.append("')\"");
        //return "\"('" + subTokens.stream().map(subToken -> subToken.text).collect(Collectors.joining("', '")) + "')\"";
        return sb.toString();
    }


}
