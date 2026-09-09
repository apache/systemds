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

package org.apache.sysds.test.component.context;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.util.Arrays;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContextFactory;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.junit.Test;

@net.jcip.annotations.NotThreadSafe
public class SparkContextReferenceCountTest {

	/**
	 * Two DML executions sharing the JVM-wide singleton spark context (as happens
	 * with surefire parallel tests, threadCount&gt;1). When the first execution
	 * finishes and calls close(), the shared context must stay alive because the
	 * second execution still has in-flight work. Before reference counting,
	 * close() stopped the context unconditionally, which cancelled the second
	 * execution's spark job and wedged it until the test watchdog.
	 */
	@Test
	public void closeKeepsContextAliveWhileAnotherExecutionIsActive() {
		SparkExecutionContext ecA = null;
		SparkExecutionContext ecB = null;
		try {
			// execution A: create the context then register (as in DMLScript.execute)
			ecA = ExecutionContextFactory.createSparkExecutionContext();
			SparkExecutionContext.enterSparkExecution();
			JavaSparkContext sc = ecA.getSparkContext();

			// execution B is a second concurrent user with its own context instance,
			// sharing the same JVM-wide singleton spark context
			ecB = ExecutionContextFactory.createSparkExecutionContext();
			SparkExecutionContext.enterSparkExecution();

			// B's in-flight work
			JavaRDD<Integer> rdd = sc.parallelize(Arrays.asList(1, 2, 3, 4));

			// A finishes first: release its registration; close() must NOT stop the
			// context that B still uses
			SparkExecutionContext.exitSparkExecution();
			ecA.close();
			assertFalse("shared context must stay alive while another execution is active",
				sc.sc().isStopped());
			assertEquals("B's job must still run on the live context",
				10L, rdd.reduce(Integer::sum).longValue());

			// B finishes last: releasing the final registration lets close() stop it
			SparkExecutionContext.exitSparkExecution();
			ecB.close();
			assertTrue("shared context must be stopped once the last execution closes",
				sc.sc().isStopped());
		}
		finally {
			// drain any remaining registrations and stop the context so a failed
			// assertion cannot leak ref-count state into other tests in this JVM
			// (exit/close are clamped and no-op once already drained/stopped)
			SparkExecutionContext.exitSparkExecution();
			SparkExecutionContext.exitSparkExecution();
			if(ecA != null)
				ecA.close();
		}
	}

	/**
	 * An unpaired close() (a caller that borrows the shared context but never
	 * registered via enterSparkExecution()) must not stop a context another
	 * execution still uses. This fails on the old unconditional-stop code, which
	 * tore the context down out from under the active execution.
	 */
	@Test
	public void unpairedCloseDoesNotStopAContextStillInUse() {
		SparkExecutionContext active = null;
		SparkExecutionContext unregistered = null;
		try {
			// a registered, in-flight execution holds the shared context
			active = ExecutionContextFactory.createSparkExecutionContext();
			SparkExecutionContext.enterSparkExecution();
			JavaSparkContext sc = active.getSparkContext();

			// a context that never registered closes (e.g. a caller that only
			// borrows the shared context): close() must not stop a context in use
			unregistered = ExecutionContextFactory.createSparkExecutionContext();
			unregistered.close();
			assertFalse("unpaired close() must not stop a context still in use",
				sc.sc().isStopped());

			// the registered execution finishing stops the context as the last user
			SparkExecutionContext.exitSparkExecution();
			active.close();
			assertTrue("context must stop once the last registered execution closes",
				sc.sc().isStopped());
		}
		finally {
			SparkExecutionContext.exitSparkExecution();
			if(active != null)
				active.close();
			if(unregistered != null)
				unregistered.close();
		}
	}
}
