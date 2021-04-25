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
public class OffsetTests {

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
			tests.add(new Object[] {new int[] {1, 2}, t});
			tests.add(new Object[] {new int[] {2, 142}, t});
			tests.add(new Object[] {new int[] {142, 421}, t});
			tests.add(new Object[] {new int[] {1, 1023}, t});
			tests.add(new Object[] {new int[] {1023, 1024}, t});
			tests.add(new Object[] {new int[] {1023}, t});
			tests.add(new Object[] {new int[] {0, 1, 2, 3, 4, 5}, t});
			tests.add(new Object[] {new int[] {0}, t});
			tests.add(new Object[] {new int[] {Character.MAX_VALUE, ((int) Character.MAX_VALUE) + 1}, t});
			tests.add(new Object[] {new int[] {Character.MAX_VALUE, ((int) Character.MAX_VALUE) * 2}, t});
			tests.add(new Object[] {new int[] {0, 256}, t});
			tests.add(new Object[] {new int[] {0, 254}, t});
			tests.add(new Object[] {new int[] {0, Character.MAX_VALUE}, t});
			tests.add(new Object[] {new int[] {0, ((int) Character.MAX_VALUE) + 1}, t});
			tests.add(new Object[] {new int[] {0, ((int) Character.MAX_VALUE) - 1}, t});
			tests.add(new Object[] {new int[] {0, 256 * 2}, t});
			tests.add(new Object[] {new int[] {0, 255 * 2}, t});
			tests.add(new Object[] {new int[] {0, 254 * 2}, t});
			tests.add(new Object[] {new int[] {0, 254 * 3}, t});
			tests.add(new Object[] {new int[] {0, 255 * 3}, t});
			tests.add(new Object[] {new int[] {0, 256 * 3}, t});
			tests.add(new Object[] {new int[] {255 * 3, 255 * 5}, t});
			tests.add(new Object[] {new int[] {1000000, 1000000 + 255 * 5}, t});
			tests.add(new Object[] {new int[] {100000000, 100000000 + 255 * 5}, t});
			tests.add(new Object[] {new int[] {0, 1, 2, 3, 255 * 4, 1500}, t});
			tests.add(new Object[] {new int[] {0, 1, 2, 3, 4, 5}, t});
			tests.add(new Object[] {new int[] {2458248, 2458249, 2458253, 2458254, 2458256, 2458257, 2458258, 2458262,
				2458264, 2458266, 2458267, 2458271, 2458272, 2458275, 2458276, 2458281}, t});

		}
		return tests;
	}

	@Test
	public void testConstruction() {
		try {
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
		catch(Exception e) {
			e.printStackTrace();
			throw e;
		}
	}

	public void testConstruction(AOffset o) {
		try {
			AIterator i = o.getIterator();
			for(int j = 0; j < data.length; j++) {

				if(data[j] != i.value())
					fail("incorrect result using : " + o.getClass().getSimpleName() + " expected: "
						+ Arrays.toString(data) + " but was :" + o.toString());
				if(i.hasNext())
					i.next();
			}

		}
		catch(Exception e) {
			e.printStackTrace();
			throw e;
		}
	}

}
