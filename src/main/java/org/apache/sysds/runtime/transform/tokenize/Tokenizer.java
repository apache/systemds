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
import org.apache.sysds.runtime.util.DependencyTask;
import org.apache.sysds.runtime.util.DependencyThreadPool;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;

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

    public int getNumRowsEstimate(){
        // Estimate because Count Applier has less since it only outputs each unique token once
        if(internalRepresentation != null){
            if(tokenizerApplier.isWideFormat()){
                return internalRepresentation.length;
            }else {
                if(tokenizerApplier.hasPadding())
                    return internalRepresentation.length * tokenizerApplier.getMaxTokens();
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
        tokenizerApplier.allocateInternalMeta(numDocuments);
    }

    public FrameBlock tokenize(FrameBlock in) {
        return tokenize(in, 12);
    }

    public FrameBlock tokenize(FrameBlock in, int k){
        allocateInternalRepresentation(in.getNumRows());
        // First convert to internal representation
        this.build(in, k);
        FrameBlock out = new FrameBlock(this.getSchema());
        out.ensureAllocatedColumns(getNumRowsEstimate());
        // Then convert to output representation
        return this.apply(out, k);
    }

    public FrameBlock apply(FrameBlock out, int k) {
        int lastRow = -1;
        if(k > 1){
            DependencyThreadPool pool = new DependencyThreadPool(k);
            try{
                List<DependencyTask<?>> taskList = tokenizerApplier.getApplyTasks(this.internalRepresentation, out, k);
                lastRow = (Integer) pool.submitAllAndWait(taskList).stream().map(s -> (Integer)s).max(Integer::compare).get();
            }
            catch(ExecutionException | InterruptedException e) {
                LOG.error("MT Tokenizer apply failed");
                e.printStackTrace();
            }
            pool.shutdown();

        }else{
            lastRow = tokenizerApplier.applyInternalRepresentation(this.internalRepresentation, out);
        }
        if(lastRow != out.getNumRows()){
            out = out.slice(0, lastRow - 1, 0, out.getNumColumns() - 1, null);
        }

        return out;
    }

    public List<DependencyTask<?>> getBuildTasks(FrameBlock in, int k){
        List<DependencyTask<?>> tasks = tokenizerBuilder.getTasks(in, this.internalRepresentation, k);
        List<DependencyTask<?>> applierBuildTaskList = tokenizerApplier.getBuildTasks(this.internalRepresentation, k);
        if(tasks.size() != applierBuildTaskList.size())
            throw new DMLRuntimeException("Cannot create dependencies for mismatched array sizes");
        tasks.addAll(applierBuildTaskList);
        List<List<? extends Callable<?>>> deps = new ArrayList<>(Collections.nCopies(tasks.size(), null));
        Map<Integer[], Integer[]> depMap = new HashMap<>();
        for(int i = 0; i < tasks.size() / 2; i++){
            depMap.put(new Integer[]{i+applierBuildTaskList.size(), i+applierBuildTaskList.size() + 1}, new Integer[] {i, i+1});
        }
        DependencyThreadPool.createDependencyList(tasks, depMap, deps);
        tasks = DependencyThreadPool.createDependencyTasks(tasks, deps);
        return tasks;
    }

    public void build(FrameBlock in, int k){
        tokenizerApplier.allocateInternalMeta(in.getNumRows());
        if(k > 1){
            DependencyThreadPool pool = new DependencyThreadPool(k);
            try{
                pool.submitAllAndWait(getBuildTasks(in, k));
            }
            catch(ExecutionException | InterruptedException e) {
                LOG.error("MT Tokenizer build failed");
                e.printStackTrace();
            }
            pool.shutdown();

        }else{
            tokenizerBuilder.createInternalRepresentation(in, this.internalRepresentation);
            tokenizerApplier.build(this.internalRepresentation, 0, -1);
        }
    }

}
