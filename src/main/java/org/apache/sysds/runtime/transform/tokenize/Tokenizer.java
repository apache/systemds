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
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.transform.tokenize.applier.TokenizerApplier;
import org.apache.sysds.runtime.transform.tokenize.builder.TokenizerBuilder;
import org.apache.sysds.runtime.util.DependencyThreadPool;

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.stream.Collectors;

public class Tokenizer implements Serializable {

    private static final long serialVersionUID = 7155673772374114577L;
    protected static final Log LOG = LogFactory.getLog(Tokenizer.class.getName());

    private DocumentRepresentation[] internalRepresentation = null;
    private final TokenizerBuilder tokenizerBuilder;
    private final TokenizerApplier tokenizerApplier;

    protected Tokenizer(TokenizerBuilder tokenizerBuilder, TokenizerApplier tokenizerApplier) {
        this.tokenizerBuilder = tokenizerBuilder;
        this.tokenizerApplier = tokenizerApplier;
    }

    public Types.ValueType[] getSchema() {
        return tokenizerApplier.getOutSchema();
    }

    public int getMaxNumRows(int inRows) {
        return tokenizerApplier.getMaxNumRows(inRows);
    }

    public int getNumRows(){
        if(internalRepresentation != null){
            if(tokenizerApplier.isWideFormat()){
                return internalRepresentation.length;
            }else {
                return Arrays.stream(internalRepresentation).mapToInt(doc -> doc.tokens.size()).sum();
            }
        }
        throw new DMLRuntimeException("Internal Token Representation was not computed yet. Can not get exact size.");
    }

    public long getNumCols() {
        return tokenizerApplier.getNumCols();
    }

    public void allocateInternalRepresentation(int numDocuments){
        internalRepresentation = new DocumentRepresentation[numDocuments];
    }

    public FrameBlock tokenize(FrameBlock in) {
        return tokenize(in, 1);
    }

    public FrameBlock tokenize(FrameBlock in, int k){
        allocateInternalRepresentation(in.getNumRows());
        // First convert to internal representation
        this.build(in, k);
        FrameBlock out = new FrameBlock(this.getSchema());
        out.ensureAllocatedColumns(getNumRows());
        // Then convert to output representation
        return this.apply(out, k);
        /*

        // First convert to internal representation
        List<DocumentToTokens> documentsToTokenList = tokenizerPre.tokenizePre(in);
        // Then convert to output representation
        return tokenizerPost.tokenizePost(documentsToTokenList, out);
         */
    }

    public FrameBlock apply(FrameBlock out, int k) {
        if(k > 1){
            DependencyThreadPool pool = new DependencyThreadPool(k);
            try{
                pool.submitAllAndWait(tokenizerApplier.getTasks(this.internalRepresentation, out, k));
            }
            catch(ExecutionException | InterruptedException e) {
                LOG.error("MT Tokenizer apply failed");
                e.printStackTrace();
            }
            pool.shutdown();

        }else{
            tokenizerApplier.applyInternalRepresentation(this.internalRepresentation, out);
        }
        return out;
    }

    public void build(FrameBlock in, int k){
        if(k > 1){
            DependencyThreadPool pool = new DependencyThreadPool(k);
            try{
                pool.submitAllAndWait(tokenizerBuilder.getTasks(in, this.internalRepresentation, k));
            }
            catch(ExecutionException | InterruptedException e) {
                LOG.error("MT Tokenizer build failed");
                e.printStackTrace();
            }
            pool.shutdown();

        }else{

            tokenizerBuilder.createInternalRepresentation(in, this.internalRepresentation);
        }
    }

}
