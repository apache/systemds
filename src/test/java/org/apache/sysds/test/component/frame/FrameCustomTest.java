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

package org.apache.sysds.test.component.frame;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.when;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.columns.Array;
import org.apache.sysds.runtime.frame.data.columns.StringArray;
import org.apache.sysds.runtime.frame.data.lib.FrameLibAppend;
import org.apache.sysds.runtime.frame.data.lib.FrameLibDetectSchema;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public class FrameCustomTest {
	protected static final Log LOG = LogFactory.getLog(FrameCustomTest.class.getName());

	@Test
	public void castToFrame() {
		double maxp1 = Integer.MAX_VALUE + 1.0;
		MatrixBlock mb = TestUtils.generateTestMatrixBlock(100, 100, maxp1, maxp1, 1.0, 23);
		FrameBlock f = DataConverter.convertToFrameBlock(mb);
		assertTrue(f.getSchema()[0] == ValueType.INT64);
	}

	@Test
	public void castToFrame3() {
		double maxp1 = Integer.MAX_VALUE - 1.0;
		MatrixBlock mb = TestUtils.generateTestMatrixBlock(100, 100, maxp1, maxp1, 1.0, 23);
		FrameBlock f = DataConverter.convertToFrameBlock(mb);
		assertTrue(f.getSchema()[0] == ValueType.INT32);
	}

	@Test
	public void castIntegerValue() {
		MatrixBlock mb = new MatrixBlock(10, 10, Double.parseDouble("2.572306572E9"));
		FrameBlock f = DataConverter.convertToFrameBlock(mb);
		assertTrue(f.getSchema()[0] == ValueType.INT64);
	}

	@Test
	public void castToFrame2() {
		double maxp1 = Integer.MAX_VALUE + 1.1111;
		MatrixBlock mb = TestUtils.generateTestMatrixBlock(100, 100, maxp1, maxp1, 1.0, 23);
		FrameBlock f = DataConverter.convertToFrameBlock(mb);
		assertTrue(f.getSchema()[0] == ValueType.FP64);
	}


	@Test 
	public void detectSchemaError(){
		FrameBlock f = TestUtils.generateRandomFrameBlock(10, 10, 23);
		FrameBlock spy = spy(f);
		when(spy.getColumn(anyInt())).thenThrow(new RuntimeException());

		Exception e = assertThrows(DMLRuntimeException.class, () -> FrameLibDetectSchema.detectSchema(spy, 3));

		assertTrue(e.getMessage().contains("Failed to detect schema"));
	}



	@Test 
	public void appendUniqueColNames(){
		FrameBlock a = new FrameBlock(new ValueType[]{ValueType.FP32}, new String[]{"Hi"});
		a.appendRow(new String[]{"0.2"});
		FrameBlock b = new FrameBlock(new ValueType[]{ValueType.FP32}, new String[]{"There"});
		b.appendRow(new String[]{"0.5"});

		FrameBlock c = FrameLibAppend.append(a, b, true);

		assertTrue(c.getColumnName(0).equals("Hi"));
		assertTrue(c.getColumnName(1).equals("There"));
	}


	@Test 
	public void detectSchema(){
		FrameBlock f = new FrameBlock(new Array[]{new StringArray(new String[]{"00000001", "e013af63"})});
		assertEquals("HASH32", FrameLibDetectSchema.detectSchema(f, 1).get(0,0));
	}

	@Test 
	public void detectSchema2(){
		FrameBlock f = new FrameBlock(new Array[]{new StringArray(new String[]{"10000001", "e013af63"})});
		assertEquals("HASH32", FrameLibDetectSchema.detectSchema(f, 1).get(0,0));
	}

	@Test 
	public void detectSchema3(){
		FrameBlock f = new FrameBlock(new Array[]{new StringArray(new String[]{"e013af63","10000001"})});
		assertEquals("HASH32", FrameLibDetectSchema.detectSchema(f, 1).get(0,0));
	}
}
