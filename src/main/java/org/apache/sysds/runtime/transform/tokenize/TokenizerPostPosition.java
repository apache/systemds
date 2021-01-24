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

import java.util.HashMap;
import java.util.List;

public class TokenizerPostPosition implements TokenizerPost{
    @Override
    public FrameBlock tokenizePost(HashMap<String, List<Tokenizer.Token>> tl, FrameBlock out) {
        tl.forEach((key, tokenList) -> {
            for (Tokenizer.Token token: tokenList) {
                Object[] row = {key, token.startIndex, token.textToken};
                out.appendRow(row);
            }
        });

        return out;
    }

    @Override
    public Types.ValueType[] getOutSchema() {
        // Not sure why INT64 is required here, but CP Instruction fails otherwise
        return new Types.ValueType[]{Types.ValueType.STRING, Types.ValueType.INT64, Types.ValueType.STRING};
    }
}
