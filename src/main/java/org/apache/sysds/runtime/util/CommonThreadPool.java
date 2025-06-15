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
import java.util.Collection;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentHashMap;
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
import org.apache.sysds.utils.stats.InfrastructureAnalyzer;

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

	/** A secondary thread local executor that use a custom number of threads for PARFOR */
	private static ConcurrentHashMap<Long, CommonThreadPool> shared2 = null;
	/** Dynamic thread pool, that dynamically allocate threads as tasks come in. */
	private static ExecutorService asyncPool = null;
	/** This common thread pool */
	private final ExecutorService _pool;

	/** Local variable indicating if there was a thread that was not main, and requested a thread pool */
	public static boolean incorrectPoolUse = false;

	/**
	 * Constructor of the threadPool. This is intended not to be used except for tests. Please use the static
	 * constructors.
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
		if(k <= 1) {
			LOG.warn("Invalid to create thread pool with <= one thread returning single thread executor",
				new RuntimeException());
			return new SameThreadExecutorService();
		}

		final Thread thisThread = Thread.currentThread();
		final String threadName = thisThread.getName();
		// Contains main, because we name our test threads TestRunner_main
		final boolean mainThread = threadName.contains("main");
		if(size == k && mainThread)
			return shared; // use the default thread pool if main thread and max parallelism.
		else if(mainThread || threadName.contains("PARFOR") || threadName.contains("FedExec")) {
			CommonThreadPool pool;
			if(shared2 == null) // If there is no current shared pool allocate one.
				shared2 = new ConcurrentHashMap<>();

			pool = shared2.get(thisThread.getId());
			if(pool == null) { // If there is no pool for this thread allocate one.
				pool = new CommonThreadPool(new ForkJoinPool(k));
				shared2.put(thisThread.getId(), pool);
			}

			return pool; // Return the shared pool for this parfor or main thread.
		}
		else {
			// If we are neither a main thread or parfor thread, allocate a new thread pool
			if(!incorrectPoolUse) {
				if(threadName.contains("test"))
					LOG.error("Thread from test is not correctly using pools, please modify thread name to contain 'main'",
						new RuntimeException());
				else
					LOG.warn(
						"An instruction allocated it's own thread pool indicating that some task is not properly reusing the threads.",
						new RuntimeException());
				incorrectPoolUse = true;
			}

			return Executors.newFixedThreadPool(k);

		}
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
		finally {
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
		if(asyncPool != null)
			// It is guaranteed not to be shut down because of the synchronized barrier
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
			for(Long e : shared2.keySet())
				shutdownPool(e);
			shared2 = null;
		}
	}

	/**
	 * Shutdown the asynchronous pool associated with the given thread.
	 * 
	 * @param thread The thread given that could or could not have allocated a thread pool itself.
	 */
	public synchronized static void shutdownAsyncPools(Thread thread) {
		if(shared2 != null)
			shutdownPool(thread.getId());
	}

	private static void shutdownPool(long id) {
		final CommonThreadPool p = shared2.get(id);
		if(p != null) {
			p._pool.shutdownNow();
			shared2.remove(id);
		}
	}

	/**
	 * Get if the current threads that calls have a cached thread pool available.
	 * 
	 * This method is synchronized to analyze the general cache not the common thread pool for the main thread.
	 * 
	 * @return If there is a thread pool allocated for this thread.
	 */
	public synchronized static boolean generalCached() {
		return shared2 != null && shared2.get(Thread.currentThread().getId()) != null;
	}

	/**
	 * If there is a thread pool cached for this thread.
	 * 
	 * @return if there is a cached thread pool.
	 */
	public final boolean isCached() {
		return _pool.equals(shared) || generalCached();
	}

	@Override
	public synchronized void shutdown() {
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

	/**
	 * This method returns true or false depending on if the current thread have an overhead of having to allocate it's
	 * own thread pool if parallelization is requested.
	 * 
	 * Currently if it is the Main thread or a PARFOR thread, then we suggest using parallelism since we can reuse the
	 * allocated threads.
	 * 
	 * @return If parallelism is suggested.
	 */
	public static boolean useParallelismOnThread() {
		Thread t = Thread.currentThread();
		String name = t.getName();
		if(name.equals("main"))
			return true;
		else if(name.contains("PARFOR"))
			return true;
		else if(name.contains("test"))
			return true;
		else
			return false;
	}

	public static class SameThreadExecutorService implements ExecutorService {

		private SameThreadExecutorService() {
			// private constructor.
		}

		@Override
		public void execute(Runnable command) {
			command.run();
		}

		@Override
		public void shutdown() {
			// nothing
		}

		@Override
		public List<Runnable> shutdownNow() {
			return new ArrayList<>();
		}

		@Override
		public boolean isShutdown() {
			return false;
		}

		@Override
		public boolean isTerminated() {
			return false;

		}

		@Override
		public boolean awaitTermination(long timeout, TimeUnit unit) throws InterruptedException {
			return true;
		}

		@Override
		public <T> Future<T> submit(Callable<T> task) {
			return new NonFuture<>(task);
		}

		@Override
		public <T> Future<T> submit(Runnable task, T result) {
			return new NonFuture<>(() -> {
				task.run();
				return result;
			});
		}

		@Override
		public Future<?> submit(Runnable task) {
			return new NonFuture<>(() -> {
				task.run();
				return null;
			});
		}

		@Override
		public <T> List<Future<T>> invokeAll(Collection<? extends Callable<T>> tasks) throws InterruptedException {
			List<Future<T>> ret = new ArrayList<>();
			for(Callable<T> t : tasks)
				ret.add(new NonFuture<>(t));
			return ret;
		}

		@Override
		public <T> List<Future<T>> invokeAll(Collection<? extends Callable<T>> tasks, long timeout, TimeUnit unit)
			throws InterruptedException {
			return invokeAll(tasks);
		}

		@Override
		public <T> T invokeAny(Collection<? extends Callable<T>> tasks) throws InterruptedException, ExecutionException {
			Exception e = null;
			for(Callable<T> t : tasks) {
				try {
					T r = t.call();
					return r;
				}
				catch(Exception ee) {
					e = ee;
				}

			}
			throw new ExecutionException("failed", e);
		}

		@Override
		public <T> T invokeAny(Collection<? extends Callable<T>> tasks, long timeout, TimeUnit unit)
			throws InterruptedException, ExecutionException, TimeoutException {
			Exception e = null;
			for(Callable<T> t : tasks) {
				try {
					T r = t.call();
					return r;
				}
				catch(Exception ee) {
					e = ee;
				}

			}
			throw new ExecutionException("failed", e);
		}

		private static class NonFuture<V> implements Future<V> {

			V r;

			protected NonFuture(Callable<V> c) {
				try {
					r = c.call();
				}
				catch(Exception e) {
					throw new RuntimeException(e);
				}
			}

			@Override
			public boolean cancel(boolean mayInterruptIfRunning) {
				return true;
			}

			@Override
			public boolean isCancelled() {
				return false;
			}

			@Override
			public boolean isDone() {
				return true;
			}

			@Override
			public V get() throws InterruptedException, ExecutionException {
				return r;
			}

			@Override
			public V get(long timeout, TimeUnit unit) throws InterruptedException, ExecutionException, TimeoutException {
				return r;
			}
		}
	}
}
