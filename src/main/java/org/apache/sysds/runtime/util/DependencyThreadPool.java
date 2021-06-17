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

package org.apache.sysds.runtime.util;

import org.apache.sysds.runtime.DMLRuntimeException;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

public class DependencyThreadPool{

    private final ExecutorService _pool;

    public DependencyThreadPool(int k){
        _pool = CommonThreadPool.get(k);
    }

    public List<Future<Future<?>>> submitAll(List<DependencyTask<?>> dtasks) {
        List<Future<Future<?>>> futures = new ArrayList<>();
        for(DependencyTask<?> t : dtasks){
            CompletableFuture<Future<?>> f = new CompletableFuture<>();
            t.addPool(_pool);
            if(t.isReady()){
                f.complete(_pool.submit(t));
                futures.add(f);
            }else{
                t.assignFuture(f);
                futures.add(f);
            }
        }
        return futures;
    }


    public List<Future<Future<?>>> submitAll(List<? extends Callable<?>> tasks,
                                             List<List<? extends Callable<?>>> dependencies) {
        List<DependencyTask<?>> dtasks = createDependencyTasks(tasks, dependencies);
        return submitAll(dtasks);
    }

    public List<Object> submitAllAndWait(List<DependencyTask<?>> dtasks)
            throws ExecutionException, InterruptedException {
        List<Object> res = new ArrayList<>();
        List<Future<Future<?>>> futures = submitAll(dtasks);
        for(Future<Future<?>> ff: futures){
            res.add(ff.get().get());
        }
        return res;
    }

    public static DependencyTask<?> createDependencyTask(Callable<?> task){
        return new DependencyTask<>(task, new ArrayList<>());
    }

    public static List<DependencyTask<?>> createDependencyTasks(List<? extends Callable<?>> tasks,
                                                                    List<List<? extends Callable<?>>> dependencies){
        if(dependencies != null && tasks.size() != dependencies.size())
            throw new DMLRuntimeException("Could not create DependencyTasks since the input array sizes are where mismatched");
        List<DependencyTask<?>> ret = new ArrayList<>();
        Map<Callable<?>, DependencyTask<?>> map = new HashMap<>();
        for (Callable<?> task : tasks) {
            DependencyTask<?> dt;
            if (task instanceof  DependencyTask){
                dt = (DependencyTask<?>) task;
            }else{
                dt = new DependencyTask<>(task, new ArrayList<>());
            }
            ret.add(dt);
            map.put(task, dt);
        }
        if(dependencies == null)
            return ret;

        for(int i = 0; i < tasks.size(); i++){
            List<? extends Callable<?>> deps = dependencies.get(i);
            if(deps == null)
                continue;
            DependencyTask<?> t = ret.get(i);
            for(Callable<?> dep : deps){
                DependencyTask<?> dt = map.get(dep);
                if (dt != null)
                    dt.addDependent(t);
            }
        }
        return ret;
    }
}