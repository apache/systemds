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

public class TokenizerWhitespaceBOW extends Tokenizer {

    private static final long serialVersionUID = 9130577081982055688L;

    protected TokenizerWhitespaceBOW(Types.ValueType[] schema, int[] colList) {
        super(schema, colList);
    }

    @Override
    public FrameBlock tokenize(FrameBlock in, FrameBlock out) {
        String[][] data = {
                {
                        "id1", "token1", "10"
                },
                {
                        "id1", "token2", "20"
                },
                {
                        "id2", "token2", "30"
                }
        };
        return new FrameBlock(getSchema(), data);
    }
}
