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

package org.apache.sysds.runtime.transform.encode;

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.CacheBlock;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

import static org.apache.sysds.runtime.util.UtilFunctions.getEndIndex;

public class ColumnEncoderWordEmbedding extends ColumnEncoder {
    private MatrixBlock wordEmbeddings;

    //domain size is equal to the number columns of the embedding column (equal to length of an embedding vector)
    @Override
    public int getDomainSize(){
        return wordEmbeddings.getNumColumns();
    }
    protected ColumnEncoderWordEmbedding(int colID) {
        super(colID);
    }

    @Override
    protected double getCode(CacheBlock<?> in, int row) {
        throw new NotImplementedException();
    }

    @Override
    protected double[] getCodeCol(CacheBlock<?> in, int startInd, int blkSize) {
        throw new NotImplementedException();
    }

    //previous recode replaced strings with indices of the corresponding matrix row index
    //now, the indices are replaced with actual word embedding vectors
    //current limitation: in case the transform is done on multiple cols, the same embedding
    //matrix is used for both transform
    @Override
    public void applyDense(CacheBlock<?> in, MatrixBlock out, int outputCol, int rowStart, int blk){
        if (!(in instanceof MatrixBlock)){
            throw new DMLRuntimeException("ColumnEncoderWordEmbedding called with: " + in.getClass().getSimpleName() +
                    " and not MatrixBlock");
        }
        int rowEnd = getEndIndex(in.getNumRows(), rowStart, blk);
        //map each recoded index to the corresponding embedding vector
        for(int i=rowStart; i<rowEnd; i++){
            double embeddingIndex = in.getDouble(i, outputCol);
            //fill row with zeroes
            if(Double.isNaN(embeddingIndex)){
                for (int j = outputCol; j < outputCol + getDomainSize(); j++)
                    out.quickSetValue(i, j, 0.0);
            }
            //array copy
            else{
                for (int j = outputCol; j < outputCol + getDomainSize(); j++){
                    out.quickSetValue(i, j, wordEmbeddings.quickGetValue((int) embeddingIndex - 1,j - outputCol ));
                }
            }
        }
    }


    @Override
    protected TransformType getTransformType() {
        return TransformType.WORD_EMBEDDING;
    }

    @Override
    public void build(CacheBlock<?> in) {
        throw new NotImplementedException();
    }

    @Override
    public void allocateMetaData(FrameBlock meta) {
        throw new NotImplementedException();
    }

    @Override
    public FrameBlock getMetaData(FrameBlock out) {
        throw new NotImplementedException();
    }

    @Override
    public void initMetaData(FrameBlock meta) {
        return;
    }

    //save embeddings matrix reference for apply step
    @Override
    public void initEmbeddings(MatrixBlock embeddings){
        this.wordEmbeddings = embeddings;
    }
}
