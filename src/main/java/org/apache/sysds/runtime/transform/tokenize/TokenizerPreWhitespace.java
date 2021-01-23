package org.apache.sysds.runtime.transform.tokenize;

import org.apache.sysds.runtime.matrix.data.FrameBlock;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class TokenizerPreWhitespace implements TokenizerPre {

    private static final long serialVersionUID = 539127244034913364L;

    private final String splitRegex = "\\s+";

    private int idCol;
    private int tokenizeCol;

    public TokenizerPreWhitespace(int idCol, int tokenizeCol) {
        this.idCol = idCol;
        this.tokenizeCol = tokenizeCol;
    }

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
            // Convert index value to Java (0-based) from DML (1-based)
            String key = s[idCol - 1];
            String text = s[tokenizeCol - 1];
            // Transform to Bag format internally
            List<Tokenizer.Token> tokenList = splitToTokens(text);
            documentsToTokenList.put(key, tokenList);
        });

        return documentsToTokenList;
    }
}
