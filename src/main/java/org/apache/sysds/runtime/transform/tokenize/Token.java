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

    private List<SubToken> subTokens = new ArrayList<>();

    public Token(){}

    public Token(String token, long startIndex) {
        subTokens.add(new SubToken(token, startIndex));
    }

    public Token(List<String> tokens, List<Long> startIndex){
        if(tokens.size() != startIndex.size())
            throw new DMLRuntimeException("Cannot create token from mismatched input sizes");
        subTokens = IntStream.range(0, tokens.size()).mapToObj(i -> new SubToken(tokens.get(i), startIndex.get(i))).collect(Collectors.toList());
    }

    public Token(List<Token> subList) {
        for(Token token: subList){
            subTokens.addAll(token.subTokens);
        }
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
        for(SubToken t: subTokens){
            sb.append("', '").append(t.text);
        }
        sb.append("'\"");
        //return "\"('" + subTokens.stream().map(subToken -> subToken.text).collect(Collectors.joining("', '")) + "')\"";
        return sb.toString();
    }


}
