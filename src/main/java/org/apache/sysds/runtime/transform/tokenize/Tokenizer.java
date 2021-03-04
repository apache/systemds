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
import java.util.List;

public class Tokenizer implements Serializable {

    private static final long serialVersionUID = 7155673772374114577L;
    protected static final Log LOG = LogFactory.getLog(Tokenizer.class.getName());

    private final TokenizerPre tokenizerPre;
    private final TokenizerPost tokenizerPost;

    protected Tokenizer(TokenizerPre tokenizerPre, TokenizerPost tokenizerPost) {

        this.tokenizerPre = tokenizerPre;
        this.tokenizerPost = tokenizerPost;
    }

    public Types.ValueType[] getSchema() {
        return tokenizerPost.getOutSchema();
    }

    public long getNumRows(long inRows) {
        return tokenizerPost.getNumRows(inRows);
    }

    public long getNumCols() {
        return tokenizerPost.getNumCols();
    }

    public FrameBlock tokenize(FrameBlock in, FrameBlock out) {
        // First convert to internal representation
        List<DocumentToTokens> documentsToTokenList = tokenizerPre.tokenizePre(in);
        // Then convert to output representation
        return tokenizerPost.tokenizePost(documentsToTokenList, out);
    }

    static class Token {
        String textToken;
        long startIndex;
        long endIndex;

        public Token(String token, long startIndex) {
            this.textToken = token;
            this.startIndex = startIndex;
            this.endIndex = startIndex + token.length();
        }
    }

    static class DocumentToTokens {
        List<Object> keys;
        List<Tokenizer.Token> tokens;

        public DocumentToTokens(List<Object> keys, List<Tokenizer.Token> tokens) {
            this.keys = keys;
            this.tokens = tokens;
        }
    }
}
