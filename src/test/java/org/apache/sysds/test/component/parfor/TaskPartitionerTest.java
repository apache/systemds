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

package org.apache.sysds.test.component.parfor;

import org.apache.sysds.runtime.controlprogram.ParForProgramBlock.PTaskPartitioner;
import org.apache.sysds.runtime.controlprogram.parfor.LocalTaskQueue;
import org.apache.sysds.runtime.controlprogram.parfor.Task;
import org.apache.sysds.runtime.controlprogram.parfor.TaskPartitioner;
import org.apache.sysds.runtime.controlprogram.parfor.TaskPartitionerFactory;
import org.apache.sysds.runtime.instructions.cp.IntObject;
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.utils.stats.InfrastructureAnalyzer;
import org.junit.Assert;
import org.junit.Test;

public class TaskPartitionerTest extends AutomatedTestBase{

	@Override
	public void setUp() {
		
	}
	
	@Test
	public void testNaive() {
		testTaskPartitioner(2*LocalTaskQueue.MAX_SIZE, PTaskPartitioner.NAIVE);
	}
	
	@Test
	public void testStatic() {
		testTaskPartitioner(2*LocalTaskQueue.MAX_SIZE, PTaskPartitioner.STATIC);
	}
	
	@Test
	public void testFixed() {
		testTaskPartitioner(2*LocalTaskQueue.MAX_SIZE, PTaskPartitioner.FIXED);
	}
	
	@Test
	public void testFactoring() {
		testTaskPartitioner(2*LocalTaskQueue.MAX_SIZE, PTaskPartitioner.FACTORING);
	}
	
	@Test
	public void testFactoring2() {
		testTaskPartitioner(2*LocalTaskQueue.MAX_SIZE, PTaskPartitioner.FACTORING_CMIN);
	}
	
	@Test
	public void testFactoring3() {
		testTaskPartitioner(2*LocalTaskQueue.MAX_SIZE, PTaskPartitioner.FACTORING_CMAX);
	}
	
	@Test
	public void testUnknown() {
		testTaskPartitioner(1, PTaskPartitioner.UNSPECIFIED);
	}
	
	private void testTaskPartitioner(int numTasks, PTaskPartitioner type) {
		try {
			LocalTaskQueue<Task> queue = new LocalTaskQueue<>();
			TaskPartitioner partitioner = TaskPartitionerFactory.createTaskPartitioner(
				type, new IntObject(1), new IntObject(numTasks), new IntObject(1),
				numTasks, InfrastructureAnalyzer.getLocalParallelism(), "i");
			//asynchronous task creation
			CommonThreadPool.get().submit(()->partitioner.createTasks(queue));
			if( type == PTaskPartitioner.STATIC ) {
				Thread.sleep(10);
				System.out.println(queue.toString());
			}
			//consume tasks and check serialization
			Task t = null;
			while((t = queue.dequeueTask())!=LocalTaskQueue.NO_MORE_TASKS) {
				Task ts1 = Task.parseCompactString(t.toCompactString());
				Task ts2 = Task.parseCompactString(t.toCompactString(10));
				Assert.assertEquals(t.toString(), ts1.toString());
				Assert.assertEquals(t.toString(), ts2.toString());
			}
		} catch (Exception e) {
			if( type!=PTaskPartitioner.UNSPECIFIED )
				throw new RuntimeException(e);
		}
	}
}
