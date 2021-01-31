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

import org.apache.wink.json4j.JSONObject;

import java.util.ArrayList;
import java.util.List;

public class TokenizerPostPosition implements TokenizerPost{

    public TokenizerPostPosition(JSONObject params) {
        // No configurable params yet
    }

    @Override
    public FrameBlock tokenizePost(List<Tokenizer.DocumentToTokens> tl, FrameBlock out) {
        for (Tokenizer.DocumentToTokens docToToken: tl) {
            List<Object> keys = docToToken.keys;
            List<Tokenizer.Token> tokenList = docToToken.tokens;
            for (Tokenizer.Token token: tokenList) {
                // Create a row per token
                List<Object> rowList = new ArrayList<>(keys);
                // Convert to 1-based index for DML
                rowList.add(token.startIndex + 1);
                rowList.add(token.textToken);
                Object[] row = new Object[rowList.size()];
                rowList.toArray(row);
                out.appendRow(row);
            }
        };

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
        schema[i+1] = Types.ValueType.STRING;
        return schema;
    }
}
