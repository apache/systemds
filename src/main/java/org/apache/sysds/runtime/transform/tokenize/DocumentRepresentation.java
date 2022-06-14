package org.apache.sysds.runtime.transform.tokenize;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class DocumentRepresentation {
    public List<Object> keys;
    public List<Token> tokens;

    public DocumentRepresentation(List<Object> keys, List<Token> tokens) {
        this.keys = keys;
        this.tokens = tokens;
    }

    public List<Token> getTokens() {
        return tokens;
    }


    public void splitIntoNgrams(int minGram, int maxGram){
        List<Token> ngramTokens = new ArrayList<>();
        for(int n = minGram; n <= maxGram; n++){
            for(int i = 0; i < tokens.size() - n + 1; i++){
                List<Token> subList = tokens.subList(i, i+n);
                Token token = new Token(subList);
                ngramTokens.add(token);
            }
        }
        tokens = ngramTokens;
    }
}
