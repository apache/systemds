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
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.wink.json4j.JSONObject;

public class TokenizerFactory {

    public static Tokenizer createTokenizer(String spec, Types.ValueType[] schema) {
        return createTokenizer(spec, schema, -1, -1);
    }

    public static Tokenizer createTokenizer(String spec, Types.ValueType[] schema,
                                        int minCol, int maxCol) {
        Tokenizer tokenizer = null;

        try {
            //parse transform specification
            JSONObject jSpec = new JSONObject(spec);

            String algo = jSpec.getString("algo");
            String out = jSpec.getString("out");
            int id_col = jSpec.getInt("id_col"); // TODO: multi id cols
            int tokenize_col = jSpec.getInt("tokenize_col");

            TokenizerPre tokenizerPre;
            TokenizerPost tokenizerPost;

            // Note that internal representation should be independent from output representation
            if (algo.equals("whitespace")) {
                tokenizerPre = new TokenizerPreWhitespace();
            } else {
                throw new IllegalArgumentException("Algorithm {algo=" + algo + "} is not supported.");
            }

            if (out.equals("count")) {
                tokenizerPost = new TokenizerPostCount();
            } else if (out.equals("position")) {
                tokenizerPost = new TokenizerPostPosition();
            } else {
                throw new IllegalArgumentException("Output representation {out=" + out + "} is not supported.");
            }

            Types.ValueType[] outSchema = new Types.ValueType[]{Types.ValueType.STRING, Types.ValueType.STRING, Types.ValueType.STRING};
            tokenizer = new Tokenizer(outSchema, null,  tokenizerPre, tokenizerPost);

        }
        catch(Exception ex) {
            throw new DMLRuntimeException(ex);
        }
        return tokenizer;
    }
}
