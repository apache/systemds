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
	private static CommonThreadPool shared2 = null;
	/** The number of threads used in the custom secondary executor */
	private static int shared2K = -1;
	/** Dynamic thread pool, that dynamically allocate threads as tasks come in. */
	private static ExecutorService asyncPool = null;
	/** This common thread pool */
	private final ExecutorService _pool;

	/**
	 * Constructor of the threadPool.
	 * This is intended not to be used except for tests.
	 * Please use the static constructors.
	 * 
	 * @param pool The thread pool instance to use.
	 */
	public CommonThreadPool(ExecutorService pool) {
		this._pool = pool;
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
	public synchronized static ExecutorService get(int k) {
		if(size == k)
			return shared;
		else if(Thread.currentThread().getName().equals("main")) {
			if(shared2 != null && shared2K == k)
				return shared2;
			else if(shared2 == null) {
				shared2 = new CommonThreadPool(new ForkJoinPool(k));
				shared2K = k;
				return shared2;
			}
			else
				return new CommonThreadPool(Executors.newFixedThreadPool(k));
		}
		else
			return new CommonThreadPool(Executors.newFixedThreadPool(k));
	}

	/**
	 * Get if there is a current thread pool that have the given parallelism locally.
	 * 
	 * @param k the parallelism
	 * @return If we have a cached thread pool.
	 */
	public static boolean isSharedTPThreads(int k) {
		return size == k || shared2K == k || shared2K == -1;
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
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}
		finally{
			pool.shutdown();
		}
	}

	/**
	 * Get a dynamic thread pool that allocate threads as the requests are made. This pool is intended for async remote
	 * calls that does not depend on local compute.
	 * 
	 * @return A dynamic thread pool.
	 */
	public synchronized static ExecutorService getDynamicPool() {
		if(asyncPool != null && !(asyncPool.isShutdown() || asyncPool.isTerminated()) )
			return asyncPool;
		else {
			asyncPool = Executors.newCachedThreadPool();
			return asyncPool;
		}
	}

	/**
	 * Shutdown the cached thread pools.
	 */
	public synchronized static void shutdownAsyncPools() {
		if(asyncPool != null) {
			// shutdown prefetch/broadcast thread pool
			asyncPool.shutdown();
			asyncPool = null;
		}
		if(shared2 != null) {
			// shutdown shared custom thread count pool
			shared2.shutdown();
			shared2 = null;
			shared2K = -1;
		}
	}

	public final boolean isCached() {
		return _pool.equals(shared) || this.equals(shared2);
	}

	@Override
	public void shutdown() {
		if(!isCached())
			_pool.shutdown();
	}

	@Override
	public List<Runnable> shutdownNow() {
		return !isCached() ? _pool.shutdownNow() : null;
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

	@Override
	public boolean isShutdown() {
		return isCached() || _pool.isShutdown();
	}

	@Override
	public boolean isTerminated() {
		return isCached() || _pool.isTerminated();
	}

	@Override
	public boolean awaitTermination(long timeout, TimeUnit unit) throws InterruptedException {
		return isCached() || _pool.awaitTermination(timeout, unit);
	}

	@Override
	public <T> T invokeAny(Collection<? extends Callable<T>> tasks) throws InterruptedException, ExecutionException {
		return _pool.invokeAny(tasks);
	}

	@Override
	public <T> T invokeAny(Collection<? extends Callable<T>> tasks, long timeout, TimeUnit unit)
		throws InterruptedException, ExecutionException, TimeoutException {
		return _pool.invokeAny(tasks);
	}
}
