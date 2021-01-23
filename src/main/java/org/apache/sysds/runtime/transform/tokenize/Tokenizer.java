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

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.matrix.data.FrameBlock;

import java.io.Serializable;
import java.util.HashMap;
import java.util.List;

public class Tokenizer implements Serializable {

    private static final long serialVersionUID = 7155673772374114577L;
    protected static final Log LOG = LogFactory.getLog(Tokenizer.class.getName());

    protected final Types.ValueType[] _schema;
    protected final int[] _colList;

    private final TokenizerPre tokenizerPre;
    private final TokenizerPost tokenizerPost;

    protected Tokenizer(int[] colList, TokenizerPre tokenizerPre, TokenizerPost tokenizerPost) {
        _schema = tokenizerPost.getOutSchema();
        _colList = colList;
        this.tokenizerPre = tokenizerPre;
        this.tokenizerPost = tokenizerPost;
    }

    public Types.ValueType[] getSchema() {
        return _schema;
    }

    public FrameBlock tokenize(FrameBlock in, FrameBlock out) {
        // First comment to internal representation
        DocumentsToTokenList documentsToTokenList = tokenizerPre.tokenizePre(in);
        // Then convert to output representation
        return tokenizerPost.tokenizePost(documentsToTokenList, out);
    }

    static class Token {
        String textToken;
        int startIndex;
        int endIndex;

        public Token(String token, int starItndex) {
            this.textToken = token;
            this.startIndex = starItndex;
            this.endIndex = starItndex + token.length();
        }
    }

    static class DocumentsToTokenList extends HashMap<String, List<Token>> {}
}
