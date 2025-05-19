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

package org.apache.sysds.test.gpu.multigpu;

import org.junit.AfterClass;
import org.junit.FixMethodOrder;
import org.junit.Test;
import org.junit.runners.MethodSorters;

import java.util.ArrayList;
import java.util.List;

@FixMethodOrder(MethodSorters.NAME_ASCENDING)
public class SingleGPUTest extends GPUTest {

    private static List<Double> executionTimes = new ArrayList<>();

    @Test
    public void test01_gpuTest_10k() {
        runMultiGPUsTest(false, 10000);
    }

    @Test
    public void test01_gpuTest_20k() {
        runMultiGPUsTest(false, 20000);
    }

    @Test
    public void test01_gpuTest_50k() {
        runMultiGPUsTest(false, 50000);
    }

    @Test
    public void test01_gpuTest_100k() {
        runMultiGPUsTest(false, 100000);
    }

    @Test
    public void test01_gpuTest_200k() {
        runMultiGPUsTest(false, 200000);
    }

    @Test
    public void test01_gpuTest_500k() {
        runMultiGPUsTest(false, 500000);
    }

    @Override
    protected void runMultiGPUsTest(boolean multiGPUs, int numTestImages) {
        // Train the model first
        super.runTrainingScript(multiGPUs, numTestImages);

        long startTime = System.nanoTime();
        super.runMultiGPUsTest(multiGPUs, numTestImages);
        long endTime = System.nanoTime();
        double executionTime = (endTime - startTime) / 1e9;
        executionTimes.add(executionTime);
    }

    @AfterClass
    public static void printExecutionTimes() {
        System.out.println("Execution times for each test:");
        for (int i = 0; i < executionTimes.size(); i++) {
            System.out.printf("Test %d: %.3f sec\n", i + 1, executionTimes.get(i));
        }
    }
}
