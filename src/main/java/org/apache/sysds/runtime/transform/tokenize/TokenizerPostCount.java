package org.apache.sysds.runtime.transform.tokenize;

import org.apache.sysds.runtime.matrix.data.FrameBlock;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class TokenizerPostCount implements TokenizerPost{

    private static final long serialVersionUID = 6382000606237705019L;

    @Override
    public FrameBlock tokenizePost(Tokenizer.DocumentsToTokenList tl, FrameBlock out) {
        tl.forEach((key, tokenList) -> {
            // Creating the counts for BoW
            Map<String, Long> tokenCounts = tokenList.stream().collect(Collectors.groupingBy(token -> token.textToken, Collectors.counting()));
            // Sort alphabetically and remove duplicates
            List<String> sortedTokens = tokenList.stream().map(token -> token.textToken).distinct().sorted().collect(Collectors.toList());

            for (String token: sortedTokens) {
                String count = String.valueOf(tokenCounts.get(token));
                String[] row = {key, token, count};
                out.appendRow(row);
            }
        });

        return out;
    }
}
