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

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;

/**
 * This common thread pool provides an abstraction to obtain a shared thread pool.
 * 
 * If the number of logical cores is specified a ForkJoinPool.commonPool is returned on all requests.
 * 
 * If pools of different size are requested, we create new pool instances of FixedThreadPool, Unless we currently are on
 * the main thread, Then we return a shared instance of the first requested number of cores.
 * 
 * Alternatively the class also contain a dynamic threadPool, that is intended for asynchronous long running tasks with
 * low compute overhead, such as broadcast and collect from federated workers.
 */
public class CommonThreadPool implements ExecutorService {
	/** Log object */
	protected static final Log LOG = LogFactory.getLog(CommonThreadPool.class.getName());

	/** The number of threads of the machine */
	private static final int size = InfrastructureAnalyzer.getLocalParallelism();
	/**
	 * Shared thread pool used system-wide, potentially by concurrent parfor workers
	 * 
	 * we use the ForkJoinPool.commonPool() to avoid explicit cleanup, including unnecessary initialization (e.g.,
	 * problematic in jmlc) and because this commonPool resulted in better performance than a dedicated fixed thread
	 * pool.
	 */
	private static final ExecutorService shared = ForkJoinPool.commonPool();
	/** A secondary thread local executor that use a custom number of threads */
	private static ExecutorService shared2 = null;
	/** The number of threads used in the custom secondary executor */
	private static int shared2K = -1;
	/** Dynamic thread pool, that dynamically allocate threads as tasks come in. */
	private static ExecutorService triggerRemoteOPsPool = null;
	/** This common thread pool */
	private final ExecutorService _pool;

	/**
	 * Private constructor of the threadPool.
	 * 
	 * @param pool The thread pool instance to use.
	 */
	private CommonThreadPool(ExecutorService pool) {
		_pool = pool;
	}

	/**
	 * Get the shared Executor thread pool, that have the number of threads of the host system
	 * 
	 * @return An ExecutorService
	 */
	public static ExecutorService get() {
		return shared;
	}

	/**
	 * Get a Executor thread pool, that have the number of threads specified in k.
	 * 
	 * The thread pool can be reused by other processes in the same host thread requesting another pool of the same
	 * number of threads. The executor that is guaranteed ThreadLocal except if it is number of host logical cores.
	 * 
	 * 
	 * @param k The number of threads wanted
	 * @return The executor with specified parallelism
	 */
	public static ExecutorService get(int k) {
		if(size == k)
			return shared;
		else if(Thread.currentThread().getName().equals("main")) {
			if(shared2 != null && shared2K == k)
				return shared2;
			else if(shared2 == null) {
				shared2 = new CommonThreadPool(Executors.newFixedThreadPool(k));
				shared2K = k;
				return shared2;
			}
			else {
				return Executors.newFixedThreadPool(k);
			}
		}
		else
			return Executors.newFixedThreadPool(k);
	}

	/**
	 * Get if there is a current thread pool that have the given parallelism locally.
	 * 
	 * @param k the parallelism
	 * @return If we have a cached thread pool.
	 */
	public static boolean isSharedTPThreads(int k) {
		return InfrastructureAnalyzer.getLocalParallelism() == k || shared2K == k || shared2K == -1;
	}

	/**
	 * Invoke the collection of tasks and shutdown the pool upon job termination.
	 * 
	 * @param <T>   The type of class to return from the job
	 * @param pool  The pool to execute in
	 * @param tasks The tasks to execute
	 */
	public static <T> void invokeAndShutdown(ExecutorService pool, Collection<? extends Callable<T>> tasks) {
		try {
			// execute tasks
			List<Future<T>> ret = pool.invokeAll(tasks);
			// check for errors and exceptions
			for(Future<T> r : ret)
				r.get();
			// shutdown pool
			pool.shutdown();
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}
	}

	/**
	 * Get a dynamic thread pool that allocate threads as the requests are made. This pool is intended for async remote
	 * calls that does not depend on local compute.
	 * 
	 * @return A dynamic thread pool.
	 */
	public static ExecutorService getDynamicPool() {
		if(triggerRemoteOPsPool != null)
			return triggerRemoteOPsPool;
		else {
			triggerRemoteOPsPool = Executors.newCachedThreadPool();
			return triggerRemoteOPsPool;
		}
	}

	/**
	 * Shutdown the RDD Thread pool.
	 */
	public static void shutdownAsyncRDDPool() {
		if(triggerRemoteOPsPool != null) {
			// shutdown prefetch/broadcast thread pool
			triggerRemoteOPsPool.shutdown();
			triggerRemoteOPsPool = null;
		}
	}

	private boolean isCached() {
		return _pool == shared || _pool == shared2;
	}

	@Override
	public void shutdown() {
		if(!isCached())
			_pool.shutdown();
	}

	@Override
	public List<Runnable> shutdownNow() {
		return !isCached() ? null : _pool.shutdownNow();
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

	// unnecessary methods required for API compliance
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
