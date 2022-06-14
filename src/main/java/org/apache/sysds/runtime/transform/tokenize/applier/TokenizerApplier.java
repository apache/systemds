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

package org.apache.sysds.runtime.transform.tokenize.applier;

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.transform.tokenize.DocumentRepresentation;
import org.apache.sysds.runtime.transform.tokenize.Tokenizer;
import org.apache.sysds.runtime.util.DependencyTask;
import org.apache.sysds.runtime.util.DependencyThreadPool;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;

import static org.apache.sysds.runtime.util.UtilFunctions.getBlockSizes;

public abstract class TokenizerApplier implements Serializable {

    protected final int numIdCols;
    protected final int maxTokens;
    protected final boolean wideFormat;
    public TokenizerApplier(int numIdCols, int maxTokens, boolean wideFormat){
        this.numIdCols = numIdCols;
        this.maxTokens = maxTokens;
        this.wideFormat = wideFormat;
    }

    public void applyInternalRepresentation(DocumentRepresentation[] internalRepresentation, FrameBlock out){
        applyInternalRepresentation(internalRepresentation, out, 0, -1);
    }

    abstract void applyInternalRepresentation(DocumentRepresentation[] internalRepresentation, FrameBlock out, int startRow, int blk);

    public List<DependencyTask<?>> getTasks(DocumentRepresentation[] internalRepresentation, FrameBlock out, int k) {
        int nRows = out.getNumRows();
        List<Callable<Object>> tasks = new ArrayList<>();
        int[] blockSizes = getBlockSizes(nRows, k);
        if(blockSizes.length == 1){
            tasks.add(new TokenizerApplyTask<>(this, out, internalRepresentation, 0, -1));
        }
        else {
            for(int startRow = 0, i = 0; i < blockSizes.length; startRow+=blockSizes[i], i++){
                tasks.add(new TokenizerApplyTask<>(this, out, internalRepresentation, startRow, blockSizes[i]));
            }
        }
        return DependencyThreadPool.createDependencyTasks(tasks, null);
    }

    public abstract Types.ValueType[] getOutSchema();

    public int getMaxNumRows(int inRows) {
        if (wideFormat) {
            return inRows;
        } else {
            return inRows * maxTokens;
        }
    }

    public long getNumCols() {
        return this.getOutSchema().length;
    }

    public boolean isWideFormat() {
        return wideFormat;
    }


    protected static class TokenizerApplyTask<T extends TokenizerApplier> implements Callable<Object>{

        protected final T _tokenizerApplier;
        protected final FrameBlock _output;
        protected final DocumentRepresentation[] _internalRepresentation;
        protected final int _rowStart;
        protected final int _blk;

        protected TokenizerApplyTask(T tokenizerApplier, FrameBlock out,
                                     DocumentRepresentation[] internalRepresentation,
                                     int rowStart, int blk){
            this._tokenizerApplier = tokenizerApplier;
            this._output = out;
            this._internalRepresentation = internalRepresentation;
            this._rowStart = rowStart;
            this._blk = blk;
        }

        @Override
        public Object call() throws Exception {
            this._tokenizerApplier.applyInternalRepresentation(this._internalRepresentation, this._output, this._rowStart, this._blk);
            return null;
        }
    }

}
