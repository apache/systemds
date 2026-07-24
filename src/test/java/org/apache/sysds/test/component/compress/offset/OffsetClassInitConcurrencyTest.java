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

package org.apache.sysds.test.component.compress.offset;

import static org.junit.Assert.fail;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.atomic.AtomicReference;

import org.junit.Test;

/**
 * Regression guard for a superclass/subclass class-initialization deadlock in the offset hierarchy.
 * <p>
 * {@code AOffset} previously instantiated its {@code OffsetEmpty} subclass from a {@code static final} field, so
 * {@code AOffset.<clinit>} depended on {@code OffsetEmpty} while {@code OffsetEmpty} (being a subclass) depends on
 * {@code AOffset}. Initializing the two classes from different threads at the same time deadlocked on the JVM class
 * initialization monitors -- which only happens under concurrent first-touch (e.g. parallel tests) and is invisible to
 * the JVM deadlock detector, so it hangs forever.
 * <p>
 * This test forces a fresh, concurrent first-initialization of the offset classes through a dedicated class loader and
 * fails if it does not complete promptly.
 */
public class OffsetClassInitConcurrencyTest {

	private static final String PKG = "org.apache.sysds.runtime.compress.colgroup.offset.";

	/** Classes whose static initializers participate in the (former) cycle. */
	private static final String[] INIT_TARGETS = {PKG + "AOffset", PKG + "OffsetEmpty", PKG + "OffsetChar",
		PKG + "OffsetByte", PKG + "OffsetSingle", PKG + "OffsetTwo"};

	/** Whether a class-init cycle deadlocks depends on thread timing, so repeat to make a regression reliable to catch. */
	private static final int ROUNDS = 20;

	/** A real init deadlock never resolves; a healthy round finishes in milliseconds, so this bound is generous. */
	private static final long ROUND_TIMEOUT_MS = 10000;

	@Test(timeout = 60000)
	public void concurrentFirstInitDoesNotDeadlock() throws Exception {
		for(int round = 0; round < ROUNDS; round++)
			runConcurrentInitRound(round);
	}

	private static void runConcurrentInitRound(int round) throws Exception {
		// A fresh loader per round so the offset classes initialize from scratch (rather than reusing state from
		// an earlier round or earlier test), reproducing the concurrent first-touch race.
		final ClassLoader loader = new OffsetPackageClassLoader(OffsetClassInitConcurrencyTest.class.getClassLoader());
		final CyclicBarrier startLine = new CyclicBarrier(INIT_TARGETS.length);
		final List<Thread> threads = new ArrayList<>();
		final AtomicReference<Throwable> failure = new AtomicReference<>();

		for(String target : INIT_TARGETS) {
			final Thread t = new Thread(() -> {
				try {
					startLine.await();
					// init=true forces the static initializer to run on this thread.
					Class.forName(target, true, loader);
				}
				catch(Throwable e) {
					failure.compareAndSet(null, e);
				}
			}, "init-" + target.substring(PKG.length()));
			// Daemon so a regression (deadlock) cannot keep the JVM alive after the test times out.
			t.setDaemon(true);
			threads.add(t);
			t.start();
		}

		final long deadline = System.currentTimeMillis() + ROUND_TIMEOUT_MS;
		for(Thread t : threads) {
			final long remaining = deadline - System.currentTimeMillis();
			if(remaining > 0)
				t.join(remaining);
			if(t.isAlive())
				fail("Concurrent class initialization deadlocked in round " + round + " (thread " + t.getName()
					+ " did not finish); likely a static-init cycle between AOffset and a subclass.");
		}

		if(failure.get() != null)
			fail("Concurrent class initialization failed in round " + round + ": " + failure.get());
	}

	/** Loads the offset package classes itself (delegating everything else) so they initialize fresh. */
	private static final class OffsetPackageClassLoader extends ClassLoader {
		OffsetPackageClassLoader(ClassLoader parent) {
			super(parent);
		}

		@Override
		protected Class<?> loadClass(String name, boolean resolve) throws ClassNotFoundException {
			if(!name.startsWith(PKG))
				return super.loadClass(name, resolve);
			synchronized(getClassLoadingLock(name)) {
				Class<?> c = findLoadedClass(name);
				if(c == null)
					c = defineFromParentResource(name);
				if(resolve)
					resolveClass(c);
				return c;
			}
		}

		private Class<?> defineFromParentResource(String name) throws ClassNotFoundException {
			final String path = name.replace('.', '/') + ".class";
			try(InputStream is = getParent().getResourceAsStream(path)) {
				if(is == null)
					throw new ClassNotFoundException(name);
				final byte[] b = is.readAllBytes();
				return defineClass(name, b, 0, b.length);
			}
			catch(IOException e) {
				throw new ClassNotFoundException(name, e);
			}
		}
	}
}
