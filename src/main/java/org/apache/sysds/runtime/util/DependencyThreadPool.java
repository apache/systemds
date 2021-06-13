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
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

public class DependencyThreadPool{

    private final ExecutorService _pool;

    public DependencyThreadPool(int k){
        _pool = CommonThreadPool.get(k);
    }

    public List<Future<Future<?>>> submitAll(List<? extends Callable<?>> tasks,
                                             List<List<? extends Callable<?>>> dependencies) {
        List<Future<Future<?>>> futures = new ArrayList<>();
        List<DependencyTask<?>> dtasks = createDependencyTasks(tasks, dependencies);
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
    
    public static List<DependencyTask<?>> createDependencyTasks(List<? extends Callable<?>> tasks,
                                                                    List<List<? extends Callable<?>>> dependencies){
        if(tasks.size() != dependencies.size())
            throw new DMLRuntimeException("Could not create DependencyTasks since the input array sizes are where mismatched");
        List<DependencyTask<?>> ret = new ArrayList<>();
        Map<Callable<?>, DependencyTask<?>> map = new HashMap<>();
        for (Callable<?> task : tasks) {
            DependencyTask<?> dt = new DependencyTask<>(task, new ArrayList<>());
            ret.add(dt);
            map.put(task, dt);
        }
        for(int i = 0; i < tasks.size(); i++){
            DependencyTask<?> t = ret.get(i);
            List<? extends Callable<?>> deps = dependencies.get(i);
            if(deps == null)
                continue;
            for(Callable<?> dep : deps){
                DependencyTask<?> dt = map.get(dep);
                if (dt != null)
                    dt.addDependent(t);
            }
        }
        return ret;
    }



    private static class DependencyTask<E> implements Callable<E>{

        private final Callable<E> _task;
        private final List<DependencyTask<?>> _dependantTasks;
        private CompletableFuture<Future<?>> _future;
        private int _rdy = 0;
        private ExecutorService _pool;


        protected DependencyTask(Callable<E> task, List<DependencyTask<?>> dependantTasks){
            _dependantTasks = dependantTasks;
            _task = task;
        }

        private void addPool(ExecutorService pool){
            _pool = pool;
        }

        private void assignFuture(CompletableFuture<Future<?>> f) {
            _future = f;
        }

        private boolean isReady() {
            return _rdy == 0;
        }

        private boolean decrease(){
            synchronized (this){
                _rdy -= 1;
                return isReady();
            }
        }

        private void addDependent(DependencyTask<?> dependencyTask) {
            _dependantTasks.add(dependencyTask);
            dependencyTask._rdy += 1;
        }

        @Override
        public E call() throws Exception {
            E ret = _task.call();

            _dependantTasks.forEach(t -> {
                if(t.decrease()){
                    if(_pool == null)
                        throw new DMLRuntimeException("ExecutorService was not set for DependencyTask");
                    t._future.complete(_pool.submit(t));
                }
            });

            return ret;
        }

    }


}
