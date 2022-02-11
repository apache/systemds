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

import java.util.Collection;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;

/**
 * This common thread pool provides an abstraction to obtain a shared
 * thread pool, specifically the ForkJoinPool.commonPool, for all requests
 * of the maximum degree of parallelism. If pools of different size are
 * requested, we create new pool instances of FixedThreadPool.
 */
public class CommonThreadPool implements ExecutorService
{
	//shared thread pool used system-wide, potentially by concurrent parfor workers
	//we use the ForkJoinPool.commonPool() to avoid explicit cleanup, including
	//unnecessary initialization (e.g., problematic in jmlc) and because this commonPool
	//resulted in better performance than a dedicated fixed thread pool.
	private static final int size = InfrastructureAnalyzer.getLocalParallelism();
	private static final ExecutorService shared = ForkJoinPool.commonPool();
	private final ExecutorService _pool;
	public static ExecutorService triggerRemoteOPsPool = null;

	public CommonThreadPool(ExecutorService pool) {
		_pool = pool;
	}

	public static ExecutorService get(int k) {
		return new CommonThreadPool( (size==k) ?
			shared : Executors.newFixedThreadPool(k));
	}
	
	public static <T> void invokeAndShutdown(ExecutorService pool, Collection<? extends Callable<T>> tasks) {
		try {
			//execute tasks
			List<Future<T>> ret = pool.invokeAll(tasks);
			//check for errors and exceptions
			for( Future<T> r : ret )
				r.get();
			//shutdown pool
			pool.shutdown();
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}
	}

	public static void shutdownShared() {
		shared.shutdownNow();
	}

	public static void shutdownAsyncRDDPool() {
		if (triggerRemoteOPsPool != null) {
			//shutdown prefetch/broadcast thread pool
			triggerRemoteOPsPool.shutdown();
			triggerRemoteOPsPool = null;
		}
	}

	@Override
	public void shutdown() {
		if( _pool != shared )
			_pool.shutdown();
	}

	@Override
	public List<Runnable> shutdownNow() {
		return ( _pool != shared ) ?
			_pool.shutdownNow() : null;
	}

	@Override
	public <T> List<Future<T>> invokeAll(Collection<? extends Callable<T>> tasks) throws InterruptedException {
		return _pool.invokeAll(tasks);
	}

	@Override
	public <T> List<Future<T>> invokeAll(Collection<? extends Callable<T>> tasks, long timeout, TimeUnit unit)
			throws InterruptedException {
		return _pool.invokeAll(tasks, timeout, unit);
	}
	
	@Override
	public void execute(Runnable command) {
		_pool.execute(command);
	}

	@Override
	public <T> Future<T> submit(Callable<T> task) {
		return _pool.submit(task);
	}

	@Override
	public <T> Future<T> submit(Runnable task, T result) {
		return _pool.submit(task, result);
	}

	@Override
	public Future<?> submit(Runnable task) {
		return _pool.submit(task);
	}

	
	//unnecessary methods required for API compliance
	@Override
	public boolean isShutdown() {
		throw new NotImplementedException();
	}

	@Override
	public boolean isTerminated() {
		throw new NotImplementedException();
	}

	@Override
	public boolean awaitTermination(long timeout, TimeUnit unit) throws InterruptedException {
		throw new NotImplementedException();
	}

	@Override
	public <T> T invokeAny(Collection<? extends Callable<T>> tasks) throws InterruptedException, ExecutionException {
		throw new NotImplementedException();
	}

	@Override
	public <T> T invokeAny(Collection<? extends Callable<T>> tasks, long timeout, TimeUnit unit)
			throws InterruptedException, ExecutionException, TimeoutException {
		throw new NotImplementedException();
	}
}
