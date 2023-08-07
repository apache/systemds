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
import static org.junit.Assert.assertNotEquals;

import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import org.apache.sysds.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.junit.Test;

public class ThreadPool {
	@Test
	public void testGetTheSame() {
		ExecutorService x = CommonThreadPool.get();
		ExecutorService y = CommonThreadPool.get();
		x.shutdown();
		y.shutdown();

		assertEquals(x, y);

	}

	@Test
	public void testGetSameCustomThreadCount() {
		// choosing 7 because the machine is unlikely to have 7 logical cores.
		String name = Thread.currentThread().getName();
		Thread.currentThread().setName("main");
		ExecutorService x = CommonThreadPool.get(7);
		ExecutorService y = CommonThreadPool.get(7);
		x.shutdown();
		y.shutdown();

		Thread.currentThread().setName(name);
		assertEquals(x, y);

	}

	@Test
	public void testFromOtherThread() throws InterruptedException, ExecutionException {
		ExecutorService x = CommonThreadPool.get(5);
		Future<ExecutorService> a = x.submit(() -> CommonThreadPool.get(5));
		ExecutorService y = a.get();

		assertNotEquals(x, y);
	}

	@Test
	public void testFromOtherThreadInfrastructureParallelism() throws InterruptedException, ExecutionException {
		final int k = InfrastructureAnalyzer.getLocalParallelism();
		ExecutorService x = CommonThreadPool.get(k);
		Future<ExecutorService> a = x.submit(() -> CommonThreadPool.get(k));
		ExecutorService y = a.get();
		assertEquals(x, y);
	}

}
