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

package org.apache.sysds.test.component.matrixmult;

import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.matrix.data.LibMatrixMult;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

import java.util.Random;

public class MatrixMultTransposedTest {

    // run multiple random scenarios
    @Test
    public void testCase_noTransA_TransB() {
        for(int i=0; i<10; i++) {
            runTest(false, true);
        }
    }

    @Test
    public void testCase_TransA_NoTransB() {
        for(int i=0; i<10; i++) {
            runTest(true, false);
        }
    }

    @Test
    public void testCase_TransA_TransB() {
        for(int i=0; i<10; i++) {
            runTest(true, true);
        }
    }

    private void runTest(boolean tA, boolean tB) {
        Random rand = new Random();

        // generate random dimensions between 1 and 300
        int m = rand.nextInt(300) + 1;
        int n = rand.nextInt(300) + 1;
        int k = rand.nextInt(300) + 1;


        int rowsA = tA ? k : m;
        int colsA = tA ? m : k;
        int rowsB = tB ? n : k;
        int colsB = tB ? k : n;

        MatrixBlock ma = MatrixBlock.randOperations(rowsA, colsA, 1.0, -1, 1, "uniform", 7);
        MatrixBlock mb = MatrixBlock.randOperations(rowsB, colsB, 1.0, -1, 1, "uniform", 3);

        MatrixBlock mc = new MatrixBlock(m, n, false);
        mc.allocateDenseBlock();

        DenseBlock a = ma.getDenseBlock();
        DenseBlock b = mb.getDenseBlock();
        DenseBlock c = mc.getDenseBlock();

        LibMatrixMult.matrixMultDenseDenseMM(a, b, c, tA, tB, n, k, 0, m, 0, n);

        mc.recomputeNonZeros();

        // calc true result with existing methods
        MatrixBlock ma_in = tA ? LibMatrixReorg.transpose(ma) : ma;
        MatrixBlock mb_in = tB ? LibMatrixReorg.transpose(mb) : mb;
        MatrixBlock expected = LibMatrixMult.matrixMult(ma_in, mb_in);

        // compare results
        TestUtils.compareMatrices(expected, mc, 1e-8);
    }
}
