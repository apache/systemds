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

package org.apache.sysds.test.component.frame.compress;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.frame.data.columns.ArrayFactory.FrameArrayType;
import org.apache.sysds.runtime.frame.data.compress.ArrayCompressionStatistics;
import org.junit.Test;

public class CompressedStatisticsTest {

	@Test
	public void testPrintWithSampledRows() {
		ArrayCompressionStatistics a = new ArrayCompressionStatistics(13, 13, false, ValueType.BOOLEAN, false,
			null, 0, 0, false);
		assertTrue(a.toString().contains("EstUnique"));
	}

	@Test
	public void testPrintWithSampledRowsNone() {
		ArrayCompressionStatistics a = new ArrayCompressionStatistics(13, 13, false, ValueType.BOOLEAN, false,
			null, 0, 0, false);
		assertTrue(a.toString().contains("Use:      None"));
	}

	@Test
	public void testPrintWithSampledRowsDDC() {
		ArrayCompressionStatistics a = new ArrayCompressionStatistics(13, 13, false, ValueType.BOOLEAN, false,
			FrameArrayType.DDC, 0, 0, false);
		assertTrue(a.toString().contains("Use:       DDC"));
	}

	@Test
	public void testPrintWithAllRowsRows() {
		ArrayCompressionStatistics a = new ArrayCompressionStatistics(13, 13, false, ValueType.BOOLEAN, false,
			null, 0, 0, true);
		assertTrue(a.toString().contains("Unique"));
		assertFalse(a.toString().contains("EstUnique"));
	}

	@Test(expected = Exception.class)
	public void nullType() {
		new ArrayCompressionStatistics(13, 13, false, null, false, FrameArrayType.DDC, 0, 0, true);
	}

}
