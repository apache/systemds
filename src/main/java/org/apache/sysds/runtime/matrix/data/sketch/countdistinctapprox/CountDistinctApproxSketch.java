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

package org.apache.sysds.runtime.matrix.data.sketch.countdistinctapprox;

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.sketch.MatrixSketch;
import org.apache.sysds.runtime.matrix.operators.CountDistinctOperator;
import org.apache.sysds.runtime.matrix.operators.Operator;

// Package private
abstract class CountDistinctApproxSketch implements MatrixSketch<Integer> {
    CountDistinctOperator op;

    CountDistinctApproxSketch(Operator op) {
        if (!(op instanceof CountDistinctOperator)) {
            throw new DMLRuntimeException(String.format("Cannot create %s with given operator", CountDistinctApproxSketch.class.getSimpleName()));
        }

        this.op = (CountDistinctOperator) op;

        if (this.op.getDirection() == null) {
            throw new DMLRuntimeException("No direction was set for the operator");
        }

        if (!this.op.getDirection().isRow() && !this.op.getDirection().isCol() && !this.op.getDirection().isRowCol()) {
            throw new DMLRuntimeException(String.format("Unexpected direction: %s", this.op.getDirection()));
        }
    }

    protected void validateSketchMetadata(MatrixBlock corrBlock) {
        // (nHashes, k, D) row vector
        if (corrBlock.getNumColumns() < 3 || corrBlock.getValue(0, 0) < 0 || corrBlock.getValue(0, 1) < 0
                || corrBlock.getValue(0, 2) < 0) {
            throw new DMLRuntimeException("Sketch metadata is corrupt");
        }
    }
}
