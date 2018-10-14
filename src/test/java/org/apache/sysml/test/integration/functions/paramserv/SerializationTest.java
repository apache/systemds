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

package org.apache.sysml.test.integration.functions.paramserv;

import java.util.Arrays;

import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.paramserv.ParamservUtils;
import org.apache.sysml.runtime.instructions.cp.IntObject;
import org.apache.sysml.runtime.instructions.cp.ListObject;
import org.apache.sysml.runtime.util.DataConverter;
import org.apache.sysml.runtime.util.ProgramConverter;
import org.junit.Assert;
import org.junit.Test;
import org.junit.internal.ArrayComparisonFailure;

public class SerializationTest {
	
	public static void assertArrayEquals(double[] expecteds,
			double[] actuals, double delta) throws ArrayComparisonFailure {
		Assert.assertArrayEquals(expecteds, actuals, delta);
	}
	
	public static void assertEquals(Object expected, Object actual) {
		Assert.assertEquals(expected, actual);
    }
	
	public static void assertEquals(long expected, long actual) {
		Assert.assertEquals(expected, actual);
    }

	@Test
	public void serializeUnnamedListObject() {
		MatrixObject mo1 = generateDummyMatrix(10);
		MatrixObject mo2 = generateDummyMatrix(20);
		IntObject io = new IntObject(30);
		ListObject lo = new ListObject(Arrays.asList(mo1, mo2, io));
		String serial = ProgramConverter.serializeDataObject("key", lo);
		Object[] obj = ProgramConverter.parseDataObject(serial);
		ListObject actualLO = (ListObject) obj[1];
		MatrixObject actualMO1 = (MatrixObject) actualLO.slice(0);
		MatrixObject actualMO2 = (MatrixObject) actualLO.slice(1);
		IntObject actualIO = (IntObject) actualLO.slice(2);
		assertArrayEquals(mo1.acquireRead().getDenseBlockValues(), actualMO1.acquireRead().getDenseBlockValues(), 0);
		assertArrayEquals(mo2.acquireRead().getDenseBlockValues(), actualMO2.acquireRead().getDenseBlockValues(), 0);
		assertEquals(io.getLongValue(), actualIO.getLongValue());
	}

	@Test
	public void serializeNamedListObject() {
		MatrixObject mo1 = generateDummyMatrix(10);
		MatrixObject mo2 = generateDummyMatrix(20);
		IntObject io = new IntObject(30);
		ListObject lo = new ListObject(Arrays.asList(mo1, mo2, io), Arrays.asList("e1", "e2", "e3"));

		String serial = ProgramConverter.serializeDataObject("key", lo);
		Object[] obj = ProgramConverter.parseDataObject(serial);
		ListObject actualLO = (ListObject) obj[1];
		MatrixObject actualMO1 = (MatrixObject) actualLO.slice(0);
		MatrixObject actualMO2 = (MatrixObject) actualLO.slice(1);
		IntObject actualIO = (IntObject) actualLO.slice(2);
		assertEquals(lo.getNames(), actualLO.getNames());
		assertArrayEquals(mo1.acquireRead().getDenseBlockValues(), actualMO1.acquireRead().getDenseBlockValues(), 0);
		assertArrayEquals(mo2.acquireRead().getDenseBlockValues(), actualMO2.acquireRead().getDenseBlockValues(), 0);
		assertEquals(io.getLongValue(), actualIO.getLongValue());
	}

	public static MatrixObject generateDummyMatrix(int size) {
		double[] dl = new double[size];
		for (int i = 0; i < size; i++) {
			dl[i] = i;
		}
		MatrixObject result = ParamservUtils.newMatrixObject(DataConverter.convertToMatrixBlock(dl, true));
		result.exportData();
		return result;
	}
}
