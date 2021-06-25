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

import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

public class DependencyTask<E> implements Callable<E> {

    private final Callable<E> _task;
    private final List<DependencyTask<?>> _dependantTasks;
    private CompletableFuture<Future<?>> _future;
    private int _rdy = 0;
    private ExecutorService _pool;


    public DependencyTask(Callable<E> task, List<DependencyTask<?>> dependantTasks){
        _dependantTasks = dependantTasks;
        _task = task;
    }

    public void addPool(ExecutorService pool){
        _pool = pool;
    }

    public void assignFuture(CompletableFuture<Future<?>> f) {
        _future = f;
    }

    public boolean isReady() {
        return _rdy == 0;
    }

    public Callable<E> getTask(){
        return _task;
    }

    private boolean decrease(){
        synchronized (this){
            _rdy -= 1;
            return isReady();
        }
    }

    public void addDependent(DependencyTask<?> dependencyTask) {
        _dependantTasks.add(dependencyTask);
        dependencyTask._rdy += 1;
    }

    @Override
    public E call() throws Exception {
        E ret = _task.call();
        System.out.println(_task);
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
