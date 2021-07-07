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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.runtime.compress.colgroup.offset.AIterator;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffset;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetByte;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetChar;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public class OffsetNegativeTests {

	private enum TYPE {
		BYTE, CHAR
	}

	@Parameterized.Parameter
	public int[] data;
	@Parameterized.Parameter(1)
	public TYPE type;

	@Parameters
	public static Collection<Object[]> data() {
		ArrayList<Object[]> tests = new ArrayList<>();
		// It is assumed that the input is in sorted order, all values are positive and there are no duplicates.
		for(TYPE t : TYPE.values()) {
			tests.add(new Object[] {new int[] {1, 1,}, t});
			tests.add(new Object[] {new int[] {2, 2, 2, 2}, t});
			tests.add(new Object[] {new int[] {1, 2, 3, 4, 5, 5}, t});
			tests.add(new Object[] {null, t});
			tests.add(new Object[] {new int[] {}, t});
			
		}
		return tests;
	}

	@Test(expected = Exception.class)
	public void testConstruction() {
		switch(type) {
			case BYTE:
				testConstruction(new OffsetByte(data));
				break;
			case CHAR:
				testConstruction(new OffsetChar(data));
				break;
			default:
				throw new NotImplementedException("not implemented");
		}

	}

	public void testConstruction(AOffset o) {
		AIterator i = o.getIterator();
		for(int j = 0; j < data.length; j++) {

			if(data[j] != i.value())
				fail("incorrect result using : " + o.getClass().getSimpleName() + " expected: " + Arrays.toString(data)
					+ " but was :" + o.toString());
			if(i.hasNext())
				i.next();
		}
	}

}
