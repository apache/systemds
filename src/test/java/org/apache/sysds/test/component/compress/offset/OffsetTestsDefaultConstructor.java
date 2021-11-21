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
import java.util.Collection;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffset;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetFactory;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public class OffsetTestsDefaultConstructor {
	protected static final Log LOG = LogFactory.getLog(OffsetTestsDefaultConstructor.class.getName());

	private static final long sizeTolerance = 100;

	public int[] data;
	private AOffset o;

	@Parameters
	public static Collection<Object[]> data() {
		ArrayList<Object[]> tests = new ArrayList<>();
		// It is assumed that the input is in sorted order, all values are positive and there are no duplicates.

		tests.add(new Object[] {new int[] {1, 2}});
		tests.add(new Object[] {new int[] {2, 142}});
		tests.add(new Object[] {new int[] {142, 421}});
		tests.add(new Object[] {new int[] {1, 1023}});
		tests.add(new Object[] {new int[] {1023, 1024}});
		tests.add(new Object[] {new int[] {1023}});
		tests.add(new Object[] {new int[] {0, 1, 2, 3, 4, 5}});
		tests.add(new Object[] {new int[] {0}});
		tests.add(new Object[] {new int[] {Character.MAX_VALUE, ((int) Character.MAX_VALUE) + 1}});
		tests.add(new Object[] {new int[] {Character.MAX_VALUE, ((int) Character.MAX_VALUE) * 2}});
		tests.add(new Object[] {new int[] {0, 256}});
		tests.add(new Object[] {new int[] {0, 254}});
		tests.add(new Object[] {new int[] {0, Character.MAX_VALUE}});
		tests.add(new Object[] {new int[] {0, Character.MAX_VALUE, ((int) Character.MAX_VALUE) * 2}});
		tests.add(new Object[] {new int[] {2, Character.MAX_VALUE + 2}});
		tests.add(new Object[] {new int[] {0, ((int) Character.MAX_VALUE) + 1}});
		tests.add(new Object[] {new int[] {0, ((int) Character.MAX_VALUE) - 1}});
		tests.add(new Object[] {new int[] {0, 256 * 2}});
		tests.add(new Object[] {new int[] {0, 255 * 2}});
		tests.add(new Object[] {new int[] {0, 254 * 2}});
		tests.add(new Object[] {new int[] {0, 510, 765}});
		tests.add(new Object[] {new int[] {0, 120, 230}});
		tests.add(new Object[] {new int[] {1000, 1120, 1230}});
		tests.add(new Object[] {new int[] {0, 254 * 3}});
		tests.add(new Object[] {new int[] {0, 255, 255 * 2, 255 * 3}});
		tests.add(new Object[] {new int[] {0, 255 * 2, 255 * 3}});
		tests.add(new Object[] {new int[] {0, 255 * 2, 255 * 3, 255 * 10}});
		tests.add(new Object[] {new int[] {0, 255 * 3}});
		tests.add(new Object[] {new int[] {0, 255 * 4}});
		tests.add(new Object[] {new int[] {0, 256 * 3}});
		tests.add(new Object[] {new int[] {255 * 3, 255 * 5}});
		tests.add(new Object[] {new int[] {1000000, 1000000 + 255 * 5}});
		tests.add(new Object[] {new int[] {100000000, 100000000 + 255 * 5}});
		tests.add(new Object[] {new int[] {100000000, 100001275, 100001530}});
		tests.add(new Object[] {new int[] {0, 1, 2, 3, 255 * 4, 1500}});
		tests.add(new Object[] {new int[] {0, 1, 2, 3, 4, 5}});
		tests.add(new Object[] {new int[] {2458248, 2458249, 2458253, 2458254, 2458256, 2458257, 2458258, 2458262,
			2458264, 2458266, 2458267, 2458271, 2458272, 2458275, 2458276, 2458281}});

		return tests;
	}

	public OffsetTestsDefaultConstructor(int[] data) {
		this.data = data;
		this.o = OffsetFactory.createOffset(data);
	}

	@Test
	public void testConstruction() {
		try {
			OffsetTests.compare(o, data);
		}
		catch(Exception e) {
			e.printStackTrace();
			throw e;
		}
	}

	@Test
	public void testMemoryEstimate(){
		final long est = OffsetFactory.estimateInMemorySize(data.length, data[data.length -1]);
		final long act = o.getInMemorySize();

		if(!( act <= est + sizeTolerance))
			fail("In memory is not smaller than estimate " + est + " " + act);
	}
}
