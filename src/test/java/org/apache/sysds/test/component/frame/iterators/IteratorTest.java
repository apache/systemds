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

package org.apache.sysds.test.component.frame.iterators;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.util.Arrays;

import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.iterators.IteratorFactory;
import org.apache.sysds.runtime.frame.data.iterators.RowIterator;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public class IteratorTest {

	private final FrameBlock fb1;
	private final FrameBlock fb2;

	public IteratorTest() {
		try {
			fb1 = TestUtils.generateRandomFrameBlock(10, 10, 23);
			fb2 = TestUtils.generateRandomFrameBlock(40, 30, 22);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
			throw new RuntimeException(e);
		}
	}

	@Test
	public void StringObjectStringFB1() {
		RowIterator<Object> a = IteratorFactory.getObjectRowIterator(fb1);
		RowIterator<String> b = IteratorFactory.getStringRowIterator(fb1);
		compareIterators(a, b);
	}

	@Test
	public void StringObjectStringFB2() {
		RowIterator<Object> a = IteratorFactory.getObjectRowIterator(fb2);
		RowIterator<String> b = IteratorFactory.getStringRowIterator(fb2);
		compareIterators(a, b);
	}

	@Test
	public void StringObjectStringNotEquals() {
		RowIterator<Object> a = IteratorFactory.getObjectRowIterator(fb1);
		a.next();
		RowIterator<String> b = IteratorFactory.getStringRowIterator(fb1);
		assertNotEquals(Arrays.toString(a.next()), Arrays.toString(b.next()));
	}

	@Test
	public void StringObjectStringNotEqualsFB1vsFB2() {
		RowIterator<Object> a = IteratorFactory.getObjectRowIterator(fb1);
		RowIterator<Object> b = IteratorFactory.getObjectRowIterator(fb2);
		assertNotEquals(Arrays.toString(a.next()), Arrays.toString(b.next()));
	}

	@Test
	public void compareSubRangesFB1() {
		RowIterator<Object> a = IteratorFactory.getObjectRowIterator(fb1, 1, fb1.getNumRows());
		RowIterator<Object> b = IteratorFactory.getObjectRowIterator(fb1);
		b.next();
		compareIterators(a, b);
	}

	@Test
	public void compareSubRangesFB2() {
		RowIterator<Object> a = IteratorFactory.getObjectRowIterator(fb2, 1, fb2.getNumRows());
		RowIterator<Object> b = IteratorFactory.getObjectRowIterator(fb2);
		b.next();
		compareIterators(a, b);
	}

	@Test
	public void compareSubRangesStringFB1() {
		RowIterator<String> a = IteratorFactory.getStringRowIterator(fb1, 1, fb1.getNumRows());
		RowIterator<String> b = IteratorFactory.getStringRowIterator(fb1);
		b.next();
		compareIterators(a, b);
	}

	@Test
	public void compareSubRangesStringFB2() {
		RowIterator<String> a = IteratorFactory.getStringRowIterator(fb2, 1, fb2.getNumRows());
		RowIterator<String> b = IteratorFactory.getStringRowIterator(fb2);
		b.next();
		compareIterators(a, b);
	}

	@Test
	public void iteratorObjectSelectColumns() {
		FrameBlock fb1Slice = fb1.slice(0, fb1.getNumRows() - 1, 1, fb1.getNumColumns() - 1);
		RowIterator<Object> a = IteratorFactory.getObjectRowIterator(fb1Slice);
		int[] select = new int[fb1.getNumColumns() - 1];
		for(int i = 0; i < fb1.getNumColumns() - 1; i++) {
			select[i] = i + 2;
		}
		RowIterator<Object> b = IteratorFactory.getObjectRowIterator(fb1, select);
		compareIterators(a, b);
	}

	@Test
	public void iteratorObjectSelectColumnsFB2() {
		FrameBlock fb2Slice = fb2.slice(0, fb2.getNumRows() - 1, 1, fb2.getNumColumns() - 1);
		RowIterator<Object> a = IteratorFactory.getObjectRowIterator(fb2Slice);
		int[] select = new int[fb2.getNumColumns() - 1];
		for(int i = 0; i < fb2.getNumColumns() - 1; i++) {
			select[i] = i + 2;
		}
		RowIterator<Object> b = IteratorFactory.getObjectRowIterator(fb2, select);
		compareIterators(a, b);
	}

	@Test
	public void iteratorStringSelectColumns() {
		FrameBlock fb1Slice = fb1.slice(0, fb1.getNumRows() - 1, 1, fb1.getNumColumns() - 1);
		RowIterator<String> a = IteratorFactory.getStringRowIterator(fb1Slice);
		int[] select = new int[fb1.getNumColumns() - 1];
		for(int i = 0; i < fb1.getNumColumns() - 1; i++) {
			select[i] = i + 2;
		}
		RowIterator<String> b = IteratorFactory.getStringRowIterator(fb1, select);
		compareIterators(a, b);
	}

	@Test
	public void iteratorStringSelectColumnsFB2() {
		FrameBlock fb2Slice = fb2.slice(0, fb2.getNumRows() - 1, 1, fb2.getNumColumns() - 1);
		RowIterator<String> a = IteratorFactory.getStringRowIterator(fb2Slice);
		int[] select = new int[fb2.getNumColumns() - 1];
		for(int i = 0; i < fb2.getNumColumns() - 1; i++) {
			select[i] = i + 2;
		}
		RowIterator<String> b = IteratorFactory.getStringRowIterator(fb2, select);
		compareIterators(a, b);
	}

	@Test
	public void iteratorStringSelectColumnsSubRowsFB2() {
		FrameBlock fb2Slice = fb2.slice(1, fb2.getNumRows() - 1, 1, fb2.getNumColumns() - 1);
		RowIterator<String> a = IteratorFactory.getStringRowIterator(fb2Slice);
		int[] select = new int[fb2.getNumColumns() - 1];
		for(int i = 0; i < fb2.getNumColumns() - 1; i++) {
			select[i] = i + 2;
		}
		RowIterator<String> b = IteratorFactory.getStringRowIterator(fb2, 1, fb2.getNumRows(), select);
		compareIterators(a, b);
	}

	@Test
	public void iteratorObjectSelectColumnsSubRowsFB2() {
		FrameBlock fb2Slice = fb2.slice(1, fb2.getNumRows() - 1, 1, fb2.getNumColumns() - 1);
		RowIterator<Object> a = IteratorFactory.getObjectRowIterator(fb2Slice);
		int[] select = new int[fb2.getNumColumns() - 1];
		for(int i = 0; i < fb2.getNumColumns() - 1; i++) {
			select[i] = i + 2;
		}
		RowIterator<Object> b = IteratorFactory.getObjectRowIterator(fb2, 1, fb2.getNumRows(), select);
		compareIterators(a, b);
	}

	@Test
	public void iteratorStringSelectSingleColumnSubRowsFB2() {
		FrameBlock fb2Slice = fb2.slice(1, fb2.getNumRows() - 1, 1, 1);
		RowIterator<String> a = IteratorFactory.getStringRowIterator(fb2Slice);
		int[] select = new int[fb2.getNumColumns() - 1];
		for(int i = 0; i < fb2.getNumColumns() - 1; i++) {
			select[i] = i + 2;
		}
		RowIterator<String> b = IteratorFactory.getStringRowIterator(fb2, 1, fb2.getNumRows(), 2);
		compareIterators(a, b);
	}

	@Test
	public void iteratorObjectSelectSingleColumnSubRowsFB2() {
		FrameBlock fb2Slice = fb2.slice(1, fb2.getNumRows() - 1, 1, 1);
		RowIterator<Object> a = IteratorFactory.getObjectRowIterator(fb2Slice);
		int[] select = new int[fb2.getNumColumns() - 1];
		for(int i = 0; i < fb2.getNumColumns() - 1; i++) {
			select[i] = i + 2;
		}
		RowIterator<Object> b = IteratorFactory.getObjectRowIterator(fb2, 1, fb2.getNumRows(), 2);
		compareIterators(a, b);
	}

	@Test
	public void iteratorColumnIdFB1() {
		FrameBlock fb1Slice = fb1.slice(0, fb1.getNumRows() - 1, 1, 1);
		RowIterator<String> a = IteratorFactory.getStringRowIterator(fb1Slice);
		RowIterator<String> b = IteratorFactory.getStringRowIterator(fb1, 2);
		compareIterators(a, b);
	}

	@Test
	public void iteratorColumnId() {
		FrameBlock fb2Slice = fb2.slice(0, fb2.getNumRows() - 1, 1, 1);
		RowIterator<String> a = IteratorFactory.getStringRowIterator(fb2Slice);
		RowIterator<String> b = IteratorFactory.getStringRowIterator(fb2, 2);
		compareIterators(a, b);
	}

	@Test
	public void iteratorColumnIdObjectFB1() {
		FrameBlock fb1Slice = fb1.slice(0, fb1.getNumRows() - 1, 1, 1);
		RowIterator<Object> a = IteratorFactory.getObjectRowIterator(fb1Slice);
		RowIterator<Object> b = IteratorFactory.getObjectRowIterator(fb1, 2);
		compareIterators(a, b);
	}

	@Test
	public void iteratorColumnObjectId() {
		FrameBlock fb2Slice = fb2.slice(0, fb2.getNumRows() - 1, 1, 1);
		RowIterator<String> a = IteratorFactory.getStringRowIterator(fb2Slice);
		RowIterator<String> b = IteratorFactory.getStringRowIterator(fb2, 2);
		compareIterators(a, b);
	}

	@Test
	public void iteratorWithSchema() {
		RowIterator<String> a = IteratorFactory.getStringRowIterator(fb2);
		RowIterator<Object> b = IteratorFactory.getObjectRowIterator(fb2, //
			UtilFunctions.nCopies(fb2.getNumColumns(), ValueType.STRING));
		compareIterators(a, b);
	}

	@Test(expected = DMLRuntimeException.class)
	public void invalidRange1() {
		IteratorFactory.getStringRowIterator(fb2, -1, 1);
	}

	@Test(expected = DMLRuntimeException.class)
	public void invalidRange2() {
		IteratorFactory.getStringRowIterator(fb2, 132415, 132416);
	}

	@Test(expected = DMLRuntimeException.class)
	public void invalidRange3() {
		IteratorFactory.getStringRowIterator(fb2, 13, 4);
	}

	@Test(expected = DMLRuntimeException.class)
	public void remove() {
		RowIterator<?> a = IteratorFactory.getStringRowIterator(fb2, 0, 4);
		a.remove();
	}

	private static void compareIterators(RowIterator<?> a, RowIterator<?> b) {
		while(a.hasNext()) {
			assertTrue(b.hasNext());
			assertEquals(Arrays.toString(a.next()), Arrays.toString(b.next()));
		}
	}
}
