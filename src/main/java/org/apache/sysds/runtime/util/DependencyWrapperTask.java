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

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

/*
* Abstract class for wrapping dependency tasks.
* Subclasses need to implement the "getWrappedTasks" function which returns the tasks that should be run.
* Tasks that are set to have a dependent on this task are going to have a dependency on all child tasks.
 */
public abstract class DependencyWrapperTask<E> extends DependencyTask<E> {

	private final List<Future<Future<?>>> _wrappedTaskFutures = new ArrayList<>();
	private final CompletableFuture<Void> _submitted = new CompletableFuture<>();
	private final DependencyThreadPool _pool;

	public DependencyWrapperTask(DependencyThreadPool pool) {
		super(() -> null, new ArrayList<>());
		_pool = pool;
	}

	public void addWrappedTaskFuture(Future<Future<?>> future) {
		_wrappedTaskFutures.add(future);
	}

	public List<Future<Future<?>>> getWrappedTaskFuture() throws ExecutionException, InterruptedException {
		_submitted.get();
		return _wrappedTaskFutures;
	}

	public abstract List<DependencyTask<?>> getWrappedTasks();

	@Override
	public E call() throws Exception {
		List<DependencyTask<?>> wrappedTasks = getWrappedTasks();
		// passing the dependency to the wrapped tasks.
		_dependantTasks.forEach(t -> wrappedTasks.forEach(w -> w.addDependent(t)));
		_pool.submitAll(wrappedTasks).forEach(this::addWrappedTaskFuture);
		_submitted.complete(null);
		return super.call();
	}

}
