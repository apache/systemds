package org.apache.sysds.runtime.transform.tokenize;

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.wink.json4j.JSONException;
import org.apache.wink.json4j.JSONObject;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.function.Function;
import java.util.stream.Collectors;

public class TokenizerPostHash implements TokenizerPost{

    private static final long serialVersionUID = 4763889041868044668L;
    private final Params params;
    private final int maxTokens;

    static class Params implements Serializable {

        private static final long serialVersionUID = -256069061414241795L;

        public int num_features = 1048576;  // 2^20

        public Params(JSONObject json) throws JSONException {
            if (json != null && json.has("num_features")) {
                this.num_features = json.getInt("num_features");
            }
        }
    }

    public TokenizerPostHash(JSONObject params, int maxTokens) throws JSONException {
        this.params = new Params(params);
        this.maxTokens = maxTokens;
    }

    @Override
    public FrameBlock tokenizePost(List<Tokenizer.DocumentToTokens> tl, FrameBlock out) {
        for (Tokenizer.DocumentToTokens docToToken: tl) {
            List<Object> keys = docToToken.keys;
            List<Tokenizer.Token> tokenList = docToToken.tokens;
            // Transform to hashes
            List<Integer> hashList = tokenList.stream().map(token -> token.textToken.hashCode() % params.num_features).collect(Collectors.toList());
            // Counting the hashes
            Map<Integer, Long> hashCounts = hashList.stream().collect(Collectors.groupingBy(Function.identity(), Collectors.counting()));
            // Sorted by hash
            Map<Integer, Long> sortedHashes = new TreeMap<>(hashCounts);

            int numTokens = 0;
            for (Map.Entry<Integer, Long> hashCount: sortedHashes.entrySet()) {
                if (numTokens >= maxTokens) {
                    break;
                }
                // Create a row per token
                int hash = hashCount.getKey() + 1;
                long count = hashCount.getValue();
                List<Object> rowList = new ArrayList<>(keys);
                rowList.add((long) hash);
                rowList.add(count);
                Object[] row = new Object[rowList.size()];
                rowList.toArray(row);
                out.appendRow(row);
                numTokens++;
            }
        }

        return out;
    }

    @Override
    public Types.ValueType[] getOutSchema(int numIdCols) {
        Types.ValueType[] schema = new Types.ValueType[numIdCols + 2];
        int i = 0;
        for (; i < numIdCols; i++) {
            schema[i] = Types.ValueType.STRING;
        }
        // Not sure why INT64 is required here, but CP Instruction fails otherwise
        schema[i] = Types.ValueType.INT64;
        schema[i+1] = Types.ValueType.INT64;
        return schema;
    }
}
