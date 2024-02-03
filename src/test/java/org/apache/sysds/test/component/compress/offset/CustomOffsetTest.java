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

import static org.junit.Assert.assertEquals;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffset;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffset.OffsetSliceInfo;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetFactory;
import org.junit.Test;

public class CustomOffsetTest {
	protected static final Log LOG = LogFactory.getLog(CustomOffsetTest.class.getName());

	@Test
	public void sliceE() {
		AOffset a = OffsetFactory.createOffset(new int[] {441, 1299, 14612, 16110, 18033, 18643, 18768, 25798, 32315});

		OffsetSliceInfo i = a.slice(1000, 2000);
		assertEquals(OffsetFactory.createOffset(new int[] {299}), i.offsetSlice);
	}

	@Test
	public void catOffset() {
		// test case for BWARE
		int[] in = new int[] {17689, 37830, 44395, 57282, 67605, 72565, 77890, 104114, 127762, 208612, 211534, 216942,
			223576, 239395, 245210, 265202, 269410, 301734, 302389, 302679, 302769, 303286, 303331, 303920, 304125, 304365,
			304743, 306244, 306260, 306745, 307624, 307651, 309715, 310232, 310270, 311177};

		AOffset off = OffsetFactory.createOffset(in);
		OffsetTests.compare(off, in);
	}

	@Test
	public void catOffsetSlice() {
		// test case for BWARE

		int[] in = new int[] {17689, 37830, 44395, 57282, 67605, 72565, 77890, 104114, 127762, 208612, 211534, 216942,
			223576, 239395, 245210, 265202, 269410, 301734, 302389, 302679, 302769, 303286, 303331, 303920, 304125, 304365,
			304743, 306244, 306260, 306745, 307624, 307651, 309715, 310232, 310270, 311177};

		AOffset off = OffsetFactory.createOffset(in);
		off.slice(112000, 128000); // check for no crash

	}
}
