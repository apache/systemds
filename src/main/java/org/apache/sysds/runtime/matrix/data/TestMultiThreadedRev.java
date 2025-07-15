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

package org.apache.sysds.runtime.matrix.data;


public class TestMultiThreadedRev {
    public static void main(String[] args) {
        int rows = 10000, cols = 5000;
        int numThreads = 8;

        // Create and fill the input matrix with a recognizable pattern
        MatrixBlock input = new MatrixBlock(rows, cols, false);
        input.allocateDenseBlock();
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                input.set(i, j, i * cols + j);

        MatrixBlock output = new MatrixBlock(rows, cols, false);
        output.allocateDenseBlock();

        // Call the multi-threaded rev
        LibMatrixReorg.rev(input, output, numThreads);

        // Validate: first output row == last input row, etc.
        boolean pass = true;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double expected = input.get(rows - 1 - i, j);
                double actual = output.get(i, j);
                if (expected != actual) {
                    System.err.printf("Mismatch at (%d,%d): expected %.1f, got %.1f%n", i, j, expected, actual);
                    pass = false;
                    break;
                }
            }
            if (!pass) break;
        }

        if (pass) {
            System.out.println("Multi-threaded rev() test PASSED!");
        } else {
            System.err.println("Multi-threaded rev() test FAILED!");
        }
    }
}