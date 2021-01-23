package org.apache.sysds.runtime.transform.tokenize;

import org.apache.sysds.runtime.matrix.data.FrameBlock;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class TokenizerPreWhitespace implements TokenizerPre {

    private static final long serialVersionUID = 539127244034913364L;

    private final String splitRegex = "\\s+";

    public List<Tokenizer.Token> splitToTokens(String text) {
        List<Tokenizer.Token> tokenList = new ArrayList<>();
        String[] textTokens = text.split(splitRegex);
        int curIndex = 0;
        for(String textToken: textTokens) {
            int tokenIndex = text.indexOf(textToken, curIndex);
            curIndex = tokenIndex;
            tokenList.add(new Tokenizer.Token(textToken, tokenIndex));
        }
        return tokenList;
    }

    @Override
    public Tokenizer.DocumentsToTokenList tokenizePre(FrameBlock in) {
        Tokenizer.DocumentsToTokenList documentsToTokenList = new Tokenizer.DocumentsToTokenList();

        Iterator<String[]> iterator = in.getStringRowIterator();
        iterator.forEachRemaining(s -> {
            String key = s[0];
            String text = s[1];
            // Transform to Bag format internally
            List<Tokenizer.Token> tokenList = splitToTokens(text);
            documentsToTokenList.put(key, tokenList);
        });

        return documentsToTokenList;
    }
}
