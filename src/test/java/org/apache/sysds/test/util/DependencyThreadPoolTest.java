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

package org.apache.sysds.test.util;

import org.apache.sysds.runtime.util.DependencyThreadPool;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.test.functions.transform.mt.TransformFrameBuildMultithreadedTest;
import org.junit.Assert;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

public class DependencyThreadPoolTest extends AutomatedTestBase {
    private final static String TEST_NAME = "DependencyThreadPoolTest";
    private final static String TEST_DIR = "util/";
    private final static String TEST_CLASS_DIR = TEST_DIR + TransformFrameBuildMultithreadedTest.class.getSimpleName() + "/";


    @Override
    public void setUp() {
        TestUtils.clearAssertionInformation();
        addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"y"}));
    }

    @Test
    public void testSimpleDependency() throws InterruptedException, ExecutionException {
        DependencyThreadPool pool = new DependencyThreadPool(4);
        TestObj global = new TestObj();
        TestTaskAdd task1 = new TestTaskAdd(1, 5, global);
        TestTaskMult task2 = new TestTaskMult(2, 20, global);
        List<? extends Callable<?>> tasks = Arrays.asList(task1, task2);
        List<List<? extends Callable<?>>> dependencies = new ArrayList<>();
        dependencies.add(Arrays.asList(task2));
        dependencies.add(null);
        List<Future<Future<?>>> futures = pool.submitAll(tasks, dependencies);
        for(Future<Future<?>> ff : futures){
            ff.get().get();
        }
        Assert.assertEquals(5, global.value);
    }

    @Test
    public void testMultipleDependency() throws InterruptedException, ExecutionException {
        DependencyThreadPool pool = new DependencyThreadPool(4);
        TestObj global = new TestObj();
        TestTaskMult task1 = new TestTaskMult(1, 20, global);
        TestTaskAdd task2 = new TestTaskAdd(2, 5, global);
        TestTaskMult task3 = new TestTaskMult(3, 20, global);
        TestTaskAdd task4 = new TestTaskAdd(4, 10, global);

        List<? extends Callable<?>> tasks = Arrays.asList(task1, task2, task3, task4);
        List<List<? extends Callable<?>>> dependencies = new ArrayList<>();
        dependencies.add(Arrays.asList(task2));
        dependencies.add(null);
        dependencies.add(Arrays.asList(task2));
        dependencies.add(Arrays.asList(task3, task1));
        List<Future<Future<?>>> futures = pool.submitAll(tasks, dependencies);
        for(Future<Future<?>> ff : futures){
            ff.get().get();
        }
        Assert.assertEquals(2010, global.value);
    }



    private static class TestObj{
        public int value = 0;

        private void add(int v){
            synchronized (this){
                value += v;
            }
        }
        private void mult(int v){
            synchronized (this){
                value *= v;
            }
        }
    }


    private static class TestTaskAdd implements Callable<Integer> {
        
        int _id;
        int _time;
        TestObj _global;

        public TestTaskAdd(int id, int time, TestObj global){
            _id = id;
            _time = time;
            _global = global;
        }
        
        @Override
        public Integer call() throws Exception {
            Thread.sleep(_time);
            _global.add(_time);
            return _id;
        }
    }

    private static class TestTaskMult implements Callable<Integer> {

        int _id;
        int _time;
        TestObj _global;

        public TestTaskMult(int id, int time, TestObj global){
            _id = id;
            _time = time;
            _global = global;
        }

        @Override
        public Integer call() throws Exception {
            Thread.sleep(_time);
            _global.mult(_time);
            return _id;
        }
    }

}
