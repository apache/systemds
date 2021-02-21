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

    protected final Types.ValueType[] _schema;

    private final TokenizerPre tokenizerPre;
    private final TokenizerPost tokenizerPost;

    // Variables are saved to estimate output format
    protected final int maxTokens;
    protected final boolean wideFormat;

    protected Tokenizer(int numIdCols, boolean wideFormat, int maxTokens,
                        TokenizerPre tokenizerPre, TokenizerPost tokenizerPost) {
        // Output schema is derived from specified id cols
        _schema = tokenizerPost.getOutSchema(numIdCols, wideFormat, maxTokens);
        this.wideFormat = wideFormat;
        this.maxTokens = maxTokens;
        this.tokenizerPre = tokenizerPre;
        this.tokenizerPost = tokenizerPost;
    }

    public Types.ValueType[] getSchema() {
        return _schema;
    }

    public long getNumRows(long inRows) {
        if (wideFormat) {
            return inRows;
        } else {
            return inRows * maxTokens;
        }
    }

    public long getNumCols() {
        return this.getSchema().length;
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
