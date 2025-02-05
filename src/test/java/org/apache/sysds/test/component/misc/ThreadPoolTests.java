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

package org.apache.sysds.test.component.misc;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;
import static org.mockito.Mockito.doNothing;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.log4j.spi.LoggingEvent;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.apache.sysds.test.LoggingUtils;
import org.apache.sysds.test.LoggingUtils.TestAppender;
import org.apache.sysds.utils.stats.InfrastructureAnalyzer;
import org.junit.Test;

public class ThreadPoolTests {
	protected static final Log LOG = LogFactory.getLog(ThreadPoolTests.class.getName());

	Thread.UncaughtExceptionHandler h = new Thread.UncaughtExceptionHandler() {
		@Override
		public void uncaughtException(Thread th, Throwable ex) {
			ex.printStackTrace();
			;
			fail(th.getName() + " " + ex.getMessage());
			throw new RuntimeException(ex);
		}
	};

	@Test
	public void testGetTheSame() {
		CommonThreadPool.shutdownAsyncPools();
		ExecutorService x = CommonThreadPool.get();
		ExecutorService y = CommonThreadPool.get();
		x.shutdown();
		y.shutdown();

		assertEquals(x, y);
		CommonThreadPool.shutdownAsyncPools();
		CommonThreadPool.shutdownAsyncPools();

	}

	@Test
	public void testGetSameCustomThreadCount() {
		CommonThreadPool.shutdownAsyncPools();
		// choosing 7 because the machine is unlikely to have 7 logical cores.
		String name = Thread.currentThread().getName();
		Thread.currentThread().setName("main");
		ExecutorService x = CommonThreadPool.get(7);
		ExecutorService y = CommonThreadPool.get(7);
		x.shutdown();
		y.shutdown();

		Thread.currentThread().setName(name);
		assertEquals(x, y);
		CommonThreadPool.shutdownAsyncPools();
		CommonThreadPool.shutdownAsyncPools();

	}

	@Test
	public void testGetSameCustomThreadCountExecute() throws InterruptedException, ExecutionException {
		// choosing 7 because the machine is unlikely to have 7 logical cores.
		CommonThreadPool.shutdownAsyncPools();
		String name = Thread.currentThread().getName();
		Thread.currentThread().setName("main");
		ExecutorService x = CommonThreadPool.get(7);
		ExecutorService y = CommonThreadPool.get(7);
		assertEquals(x, y);
		int v = x.submit(() -> 5).get();
		x.shutdown();
		int v2 = y.submit(() -> 5).get();
		y.shutdown();

		Thread.currentThread().setName(name);
		assertEquals(x, y);
		assertEquals(v, v2);
		CommonThreadPool.shutdownAsyncPools();
	}

	@Test
	public void testGetSameCustomThreadCountExecuteV2() throws InterruptedException, ExecutionException {
		// choosing 7 because the machine is unlikely to have 7 logical cores.
		String name = Thread.currentThread().getName();
		Thread.currentThread().setName("main");
		ExecutorService x = CommonThreadPool.get(7);
		ExecutorService y = CommonThreadPool.get(7);
		assertEquals(x, y);
		int v = x.submit(() -> 5).get();
		int v2 = y.submit(() -> 5).get();
		x.shutdown();
		y.shutdown();

		Thread.currentThread().setName(name);
		assertEquals(x, y);
		assertEquals(v, v2);
		CommonThreadPool.shutdownAsyncPools();
	}

	@Test
	public void testGetSameCustomThreadCountExecuteV3() throws InterruptedException, ExecutionException {
		// choosing 7 because the machine is unlikely to have 7 logical cores.
		String name = Thread.currentThread().getName();
		Thread.currentThread().setName("main");
		ExecutorService x = CommonThreadPool.get(7);
		ExecutorService y = CommonThreadPool.get(7);
		assertEquals(x, y);
		x.shutdown();
		y.shutdown();
		int v = x.submit(() -> 5).get();
		int v2 = y.submit(() -> 5).get();

		Thread.currentThread().setName(name);
		assertEquals(x, y);
		assertEquals(v, v2);
		CommonThreadPool.shutdownAsyncPools();
	}

	@Test
	public void testGetSameCustomThreadCountExecuteV4() throws InterruptedException, ExecutionException {
		// choosing 7 because the machine is unlikely to have 7 logical cores.
		String name = Thread.currentThread().getName();
		Thread.currentThread().setName("main");
		CommonThreadPool.shutdownAsyncPools();
		ExecutorService x = CommonThreadPool.get(5);
		ExecutorService y = CommonThreadPool.get(7);
		assertEquals(x, y);
		x.shutdown();
		int v = x.submit(() -> 5).get();
		int v2 = y.submit(() -> 5).get();
		y.shutdown();

		Thread.currentThread().setName(name);
		assertEquals(v, v2);
		CommonThreadPool.shutdownAsyncPools();
	}

	@Test
	public void testFromOtherThread() throws InterruptedException, ExecutionException {
		CommonThreadPool.shutdownAsyncPools();
		ExecutorService x = CommonThreadPool.get(5);
		Future<ExecutorService> a = x.submit(() -> CommonThreadPool.get(5));
		ExecutorService y = a.get();
		assertNotEquals(x, y);
		CommonThreadPool.shutdownAsyncPools();
	}

	@Test
	public void testFromOtherThreadInfrastructureParallelism() throws InterruptedException, ExecutionException {
		CommonThreadPool.shutdownAsyncPools();
		Thread.currentThread().setName("main");
		final int k = InfrastructureAnalyzer.getLocalParallelism();
		ExecutorService x = CommonThreadPool.get(k);
		Future<ExecutorService> a = x.submit(() -> CommonThreadPool.get(k));

		// make sure that we wait a bit to allow the other thread to spawn and start working
		Thread.sleep(100);

		ExecutorService y = a.get();
		assertNotEquals(x, y);
		CommonThreadPool.shutdownAsyncPools();
	}

	@Test
	public void dynamic() throws InterruptedException, ExecutionException {
		CommonThreadPool.shutdownAsyncPools();
		final int k = InfrastructureAnalyzer.getLocalParallelism();
		ExecutorService x = CommonThreadPool.getDynamicPool();
		Future<ExecutorService> a = x.submit(() -> CommonThreadPool.get(k));
		ExecutorService y = a.get();
		assertNotEquals(x, y);
		CommonThreadPool.shutdownAsyncPools();
	}

	@Test
	public void dynamicSame() throws InterruptedException, ExecutionException {
		CommonThreadPool.shutdownAsyncPools();
		ExecutorService x = CommonThreadPool.getDynamicPool();
		ExecutorService y = CommonThreadPool.getDynamicPool();
		assertEquals(x, y);
		CommonThreadPool.shutdownAsyncPools();
	}

	@Test
	public void justWorks() throws InterruptedException, ExecutionException {

		String name = Thread.currentThread().getName();
		Thread.currentThread().setName("main");
		for(int j = 0; j < 2; j++) {
			for(int i = 4; i < 17; i++) {
				ExecutorService p = CommonThreadPool.get(i);
				final Integer l = i;
				assertEquals(l, p.submit(() -> l).get());
				p.shutdown();
			}
		}
		Thread.currentThread().setName(name);
	}

	@Test
	public void justWorksNotMain() throws InterruptedException, ExecutionException {

		for(int j = 0; j < 2; j++) {

			for(int i = 4; i < 10; i++) {
				ExecutorService p = CommonThreadPool.get(i);
				final Integer l = i;
				assertEquals(l, p.submit(() -> l).get());
				p.shutdown();

			}
		}
	}

	@Test
	public void justWorksShutdownNow() throws InterruptedException, ExecutionException {

		String name = Thread.currentThread().getName();
		Thread.currentThread().setName("main");
		for(int j = 0; j < 2; j++) {

			for(int i = 4; i < 16; i++) {
				ExecutorService p = CommonThreadPool.get(i);
				final Integer l = i;
				assertEquals(l, p.submit(() -> l).get());
				p.shutdownNow();

			}
		}
		Thread.currentThread().setName(name);
	}

	@Test
	public void justWorksShutdownNowNotMain() throws InterruptedException, ExecutionException {

		Thread t = new Thread(() -> {
			for(int j = 0; j < 2; j++) {

				for(int i = 4; i < 16; i++) {
					ExecutorService p = CommonThreadPool.get(i);
					final Integer l = i;
					try {
						assertEquals(l, p.submit(() -> l).get());
					}
					catch(Exception e) {

					}
					finally {
						p.shutdown();
					}

				}
			}
		}, "somethingOtherThanMM");
		t.start();
		t.join();
	}

	@Test
	public void mock1() throws NoSuchFieldException, SecurityException, IllegalArgumentException, IllegalAccessException,
		InterruptedException, ExecutionException, TimeoutException {

		ExecutorService p = mock(ExecutorService.class);
		ExecutorService c = new CommonThreadPool(p);

		when(p.shutdownNow()).thenReturn(null);
		assertNull(c.shutdownNow());

		Collection<Callable<Integer>> cc = null;
		when(p.invokeAll(cc)).thenReturn(null);
		assertNull(c.invokeAll(cc));
		when(p.invokeAll(cc, 1L, TimeUnit.DAYS)).thenReturn(null);
		assertNull(c.invokeAll(cc, 1, TimeUnit.DAYS));
		doNothing().when(p).execute((Runnable) null);
		c.execute((Runnable) null);

		when(p.submit((Callable<Integer>) null)).thenReturn(null);
		assertNull(c.submit((Callable<Integer>) null));

		when(p.submit((Runnable) null, null)).thenReturn(null);
		assertNull(c.submit((Runnable) null, null));
		// when(tp.pool()).thenReturn(p);

		when(p.submit((Runnable) null)).thenReturn(null);
		assertNull(c.submit((Runnable) null));

		when(p.isShutdown()).thenReturn(false);
		assertFalse(c.isShutdown());
		when(p.isShutdown()).thenReturn(true);
		assertTrue(c.isShutdown());

		when(p.isTerminated()).thenReturn(false);
		assertFalse(c.isTerminated());
		when(p.isTerminated()).thenReturn(true);
		assertTrue(c.isTerminated());

		when(p.awaitTermination(10, TimeUnit.DAYS)).thenReturn(false);
		assertFalse(c.awaitTermination(10, TimeUnit.DAYS));
		when(p.awaitTermination(10, TimeUnit.DAYS)).thenReturn(true);
		assertTrue(c.awaitTermination(10, TimeUnit.DAYS));

		when(p.invokeAny(cc)).thenReturn(null);
		assertNull(c.invokeAny(cc));
		when(p.invokeAny(cc, 1L, TimeUnit.DAYS)).thenReturn(null);
		assertNull(c.invokeAny(cc, 1, TimeUnit.DAYS));
		doNothing().when(p).execute((Runnable) null);
		c.execute((Runnable) null);

	}

	@Test
	public void mock2() throws NoSuchFieldException, SecurityException, IllegalArgumentException, IllegalAccessException,
		InterruptedException, ExecutionException, TimeoutException {

		CommonThreadPool p = mock(CommonThreadPool.class);
		when(p.isShutdown()).thenCallRealMethod();
		when(p.isTerminated()).thenCallRealMethod();
		when(p.awaitTermination(10, TimeUnit.DAYS)).thenCallRealMethod();
		when(p.isCached()).thenReturn(true);
		assertTrue(p.isShutdown());
		assertTrue(p.isTerminated());
		assertTrue(p.awaitTermination(10, TimeUnit.DAYS));
	}

	@Test
	public void coverEdge() {
		ExecutorService a = CommonThreadPool.get(InfrastructureAnalyzer.getLocalParallelism());
		assertTrue(new CommonThreadPool(a).isCached());
	}

	@Test(expected = DMLRuntimeException.class)
	public void invokeAndShutdownException() throws InterruptedException {
		ExecutorService p = mock(ExecutorService.class);
		ExecutorService c = new CommonThreadPool(p);
		when(p.invokeAll(null)).thenThrow(new RuntimeException("Test"));
		CommonThreadPool.invokeAndShutdown(c, null);
	}

	@Test
	public void invokeAndShutdown() throws InterruptedException {
		ExecutorService p = mock(ExecutorService.class);
		ExecutorService c = new CommonThreadPool(p);
		Collection<Callable<Integer>> cc = null;
		when(p.invokeAll(cc)).thenReturn(new ArrayList<Future<Integer>>());
		CommonThreadPool.invokeAndShutdown(c, null);
	}

	@Test
	@SuppressWarnings("all")
	public void invokeAndShutdownV2() throws InterruptedException {
		ExecutorService p = mock(ExecutorService.class);
		ExecutorService c = new CommonThreadPool(p);
		Collection<Callable<Integer>> cc = (Collection<Callable<Integer>>) null;
		List<Future<Integer>> f = new ArrayList<Future<Integer>>();
		f.add(mock(FI.class));
		when(p.invokeAll(cc)).thenReturn(f);
		CommonThreadPool.invokeAndShutdown(c, null);
	}

	private interface FI extends Future<Integer> {
	}

	ExecutorService a;

	@Test
	public void parforTest() throws Exception {
		Thread t = new Thread(() -> {
			assertTrue(CommonThreadPool.useParallelismOnThread());
			a = CommonThreadPool.get(3);
		}, "PARFOR_T1");

		t.start();
		t.join();
		ExecutorService aa = a;
		CommonThreadPool.shutdownAsyncPools(t);
		CommonThreadPool.shutdownAsyncPools(t);
		t = new Thread(() -> {
			a = CommonThreadPool.get(3);
		}, "PARFOR_T2");

		t.start();
		t.join();
		ExecutorService bb = a;
		assertNotEquals(aa, bb);
		CommonThreadPool.shutdownAsyncPools(t);
	}

	@Test
	public void notParallelismThread() throws Exception {
		Thread t = new Thread(() -> {
			assertFalse(CommonThreadPool.useParallelismOnThread());
		}, "PR_T1");
		t.start();
		t.join();
		CommonThreadPool.shutdownAsyncPools(t);
	}

	@Test
	public void ParallelismThread() throws Exception {
		Thread t = new Thread(() -> {
			assertTrue(CommonThreadPool.useParallelismOnThread());
		}, "main");
		t.start();
		t.join();
		CommonThreadPool.shutdownAsyncPools(t);
	}

	@Test
	public void ParallelismThread_test() throws Exception {
		Thread t = new Thread(() -> {
			assertTrue(CommonThreadPool.useParallelismOnThread());
		}, "fdsfasdftestfdsfa");
		t.start();
		t.join();
		CommonThreadPool.shutdownAsyncPools(t);
	}

	@Test
	public void get1ThreadPool() {
		ExecutorService e = CommonThreadPool.get(1);
		assertTrue(e instanceof CommonThreadPool.SameThreadExecutorService);
	}

	@Test
	public void get1ThreadPoolWorks() throws Exception {
		final TestAppender appender = LoggingUtils.overwrite();
		ExecutorService e = CommonThreadPool.get(1);
		Future<?> f = e.submit(() -> {
			return null;
		});
		;
		assertTrue(f.cancel(true));
		assertFalse(f.isCancelled());
		assertTrue(f.isDone());
		e.shutdown();// does nothing
		assertNull(f.get());
		assertNull(f.get(132, TimeUnit.DAYS));

		e.execute(() -> {
		}); // nothing ...

		assertTrue(e.shutdownNow().isEmpty());
		assertFalse(e.isShutdown());
		assertFalse(e.isTerminated());
		assertTrue(e.awaitTermination(0, null));

		Runnable t = new Runnable() {
			@Override
			public void run() {
				return;
			}

		};
		Future<?> r = e.submit(t, new Object());
		assertTrue(r.isDone());
		Future<?> r2 = e.submit(t);
		assertTrue(r2.isDone());
		LoggingUtils.reinsert(appender);
	}

	@Test
	public void getThreadPoolContainingTests() throws Exception {
		CommonThreadPool.incorrectPoolUse = false;
		final TestAppender appender = LoggingUtils.overwrite();
		ExecutorService pool = Executors.newFixedThreadPool(2);
		try {

			pool.submit(() -> {
				Thread.currentThread().setName("BAAAAtest");
				ExecutorService p = CommonThreadPool.get(2);
				try {
					assertTrue(p instanceof ThreadPoolExecutor);
					return null;
				}
				catch(Exception e) {
					throw e;
				}
				finally {
					p.shutdown();
				}
			}).get();

		}
		finally {

			pool.shutdown();

			for(LoggingEvent l : LoggingUtils.reinsert(appender)) {
				if(l.getLevel() == Level.ERROR)
					return;
			}
			fail("not correctly logged");
		}

	}

	@Test
	public void getThreadPoolContainingNoTests() throws Exception {
		CommonThreadPool.incorrectPoolUse = false;
		final TestAppender appender = LoggingUtils.overwrite();
		Logger.getLogger(CommonThreadPool.class).setLevel(Level.TRACE);
		ExecutorService pool = Executors.newFixedThreadPool(2);
		try {

			pool.submit(() -> {
				Thread.currentThread().setName("BAAAANoTTTessst");
				ExecutorService p = CommonThreadPool.get(2);
				try {
					assertTrue(p instanceof ThreadPoolExecutor);
					return null;
				}
				catch(Exception e) {
					throw e;
				}
				finally {
					p.shutdown();
				}
			}).get();

		}
		finally {

			pool.shutdown();

			Logger.getLogger(CommonThreadPool.class).setLevel(Level.ERROR);
			for(LoggingEvent l : LoggingUtils.reinsert(appender)) {
				if(l.getLevel() == Level.WARN)
					return;
			}
			fail("not correctly logged");
		}

	}

	@Test
	public void getThreadLocalSharedPoolsTests() throws Exception {
		CommonThreadPool.incorrectPoolUse = false;
		Thread[] ts = new Thread[10];
		for(int i = 0; i < 10; i++) {

			ts[i] = new Thread(() -> {
				ExecutorService pool = CommonThreadPool.get(2);
				try {
					assertTrue(pool instanceof CommonThreadPool);
					pool.submit(() -> {
						try {
							Thread.sleep(3000);
						}
						catch(Exception e) {
							throw new RuntimeException(e);
						}
					});
				}
				catch(Exception e) {
					throw new RuntimeException(e);
				}
				finally {
					pool.shutdown();
				}
			}, "PARFOR_" + i);

		}

		for(Thread t : ts) {
			t.setUncaughtExceptionHandler(h);
			t.start();
		}

		Thread.sleep(20);
		CommonThreadPool.shutdownAsyncPools();

		for(Thread t : ts) {
			t.join();
		}
	}

	@Test(expected = RuntimeException.class)
	public void get1ThreadPoolException() throws Exception {
		ExecutorService pool = CommonThreadPool.get(1);

		pool.submit(() -> {
			throw new RuntimeException();
		}).get();

	}

	@Test
	public void generalCached() {
		CommonThreadPool.shutdownAsyncPools();
		assertFalse(CommonThreadPool.generalCached());

		ExecutorService pool = Executors.newFixedThreadPool(2);

		try {

			pool.submit(() -> {
				assertFalse(CommonThreadPool.generalCached());
				Thread.currentThread().setName("someThingWith_main");
				ExecutorService e = CommonThreadPool.get(3);
				assertTrue(CommonThreadPool.generalCached());
				CommonThreadPool.shutdownAsyncPools(Thread.currentThread());
				assertFalse(CommonThreadPool.generalCached());
				e.shutdown();
			});

		}
		finally {
			pool.shutdown();
		}

	}

}
