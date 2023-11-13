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
import static org.mockito.Mockito.doNothing;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.junit.Test;

public class ThreadPool {
	protected static final Log LOG = LogFactory.getLog(ThreadPool.class.getName());

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
		assertNotEquals(x, y);
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
		final int k = InfrastructureAnalyzer.getLocalParallelism();
		ExecutorService x = CommonThreadPool.get(k);
		Future<ExecutorService> a = x.submit(() -> CommonThreadPool.get(k));
		ExecutorService y = a.get();
		assertEquals(x, y);
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
	public void isSharedTPThreads() throws InterruptedException, ExecutionException {
		CommonThreadPool.shutdownAsyncPools();
		for(int i = 0; i < 10; i++)
			assertTrue(CommonThreadPool.isSharedTPThreads(i));

		CommonThreadPool.shutdownAsyncPools();
	}

	@Test
	public void isSharedTPThreadsCommonSize() throws InterruptedException, ExecutionException {
		CommonThreadPool.shutdownAsyncPools();
		assertTrue(CommonThreadPool.isSharedTPThreads(InfrastructureAnalyzer.getLocalParallelism()));
		CommonThreadPool.shutdownAsyncPools();
	}

	@Test
	public void isSharedTPThreadsFalse() throws InterruptedException, ExecutionException {
		CommonThreadPool.shutdownAsyncPools();
		String name = Thread.currentThread().getName();
		Thread.currentThread().setName("main");
		CommonThreadPool.get(18);
		for(int i = 1; i < 10; i++)
			if(i != InfrastructureAnalyzer.getLocalParallelism())
				assertFalse("" + i, CommonThreadPool.isSharedTPThreads(i));
		assertTrue(CommonThreadPool.isSharedTPThreads(18));
		assertFalse(CommonThreadPool.isSharedTPThreads(19));

		Thread.currentThread().setName(name);
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

		for(int j = 0; j < 2; j++) {

			for(int i = 4; i < 16; i++) {
				ExecutorService p = CommonThreadPool.get(i);
				final Integer l = i;
				assertEquals(l, p.submit(() -> l).get());
				p.shutdownNow();

			}
		}
	}

	@Test
	public void mock1() throws NoSuchFieldException, SecurityException, IllegalArgumentException,
		IllegalAccessException, InterruptedException, ExecutionException, TimeoutException {

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
	public void mock2() throws NoSuchFieldException, SecurityException, IllegalArgumentException,
		IllegalAccessException, InterruptedException, ExecutionException, TimeoutException {

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
}
