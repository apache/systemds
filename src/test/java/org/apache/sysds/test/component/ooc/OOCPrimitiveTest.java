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

package org.apache.sysds.test.component.ooc;

import java.util.List;

import org.apache.sysds.runtime.instructions.ooc.SubscribableTaskQueue;
import org.apache.sysds.runtime.ooc.planning.OOCAccessPattern;
import org.apache.sysds.runtime.ooc.primitives.OOCPrimitive;
import org.apache.sysds.runtime.ooc.stream.FilteredOOCStream;
import org.junit.Assert;
import org.junit.Test;

public class OOCPrimitiveTest {
	@Test
	public void testGraphPatternsAndExecution() {
		TestPrimitive source = new TestPrimitive(List.of());
		TestPrimitive sink = new TestPrimitive(List.of(source, source));

		Assert.assertEquals(List.of(source), sink.getChildren());
		Assert.assertEquals(List.of(sink), source.getParents());
		source.inferPatterns();
		Assert.assertEquals(OOCAccessPattern.ANY, source.getAccessPattern());
		Assert.assertEquals(OOCAccessPattern.ANY, sink.getAccessPattern());
		Assert.assertEquals(OOCAccessPattern.COL_MAJOR, OOCAccessPattern.ROW_MAJOR.transposed());
		Assert.assertEquals(OOCAccessPattern.UNKNOWN, OOCAccessPattern.ROW_MAJOR.fused(OOCAccessPattern.COL_MAJOR));

		SubscribableTaskQueue<Integer> stream = new SubscribableTaskQueue<>();
		stream.assignPrimitive(sink);
		FilteredOOCStream<Integer> filtered = new FilteredOOCStream<>(stream, ignored -> true);
		Assert.assertSame(sink, filtered.getPrimitive());
		stream.start();
		filtered.start();
		Assert.assertTrue(sink.hasStartedExecution());
		Assert.assertEquals(1, sink._executions);
	}

	private static final class TestPrimitive extends OOCPrimitive {
		private int _executions;

		private TestPrimitive(List<OOCPrimitive> children) {
			super(children);
		}

		@Override
		protected void startExecution() {
			_executions++;
		}

		@Override
		public void inferPatterns() {
			_pattern = OOCAccessPattern.ANY;
			inferParentPatterns();
		}

		@Override
		public void requestPattern(OOCAccessPattern accessPattern) {
			_pattern = accessPattern;
		}
	}
}
