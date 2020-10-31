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

package org.apache.sysds.test.component.paramserv;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.ObjectInput;
import java.io.ObjectOutputStream;
import java.io.ObjectInputStream;
import java.util.Arrays;
import java.util.Collection;

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.paramserv.ParamservUtils;
import org.apache.sysds.runtime.instructions.cp.IntObject;
import org.apache.sysds.runtime.instructions.cp.ListObject;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.runtime.util.ProgramConverter;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

@RunWith(Parameterized.class)
public class SerializationTest {
	private int _named;

	@Parameterized.Parameters
	public static Collection<?> named() {
		return Arrays.asList(new Object[][] {{ 0 }, { 1 }});
	}

	public SerializationTest(Integer named) {
		this._named = named;
	}

	@Test
	public void serializeListObject() {
		MatrixObject mo1 = generateDummyMatrix(10);
		MatrixObject mo2 = generateDummyMatrix(20);
		IntObject io = new IntObject(30);
		ListObject lot = new ListObject(Arrays.asList(mo2));
		ListObject lo;

		if (_named == 1)
			 lo = new ListObject(Arrays.asList(mo1, lot, io), Arrays.asList("e1", "e2", "e3"));
		else
			lo = new ListObject(Arrays.asList(mo1, lot, io));

		ListObject loDeserialized = null;

		// serialize and back
		try {
			ByteArrayOutputStream bos = new ByteArrayOutputStream();
			ObjectOutputStream out = new ObjectOutputStream(bos);
			out.writeObject(lo);
			out.flush();
			byte[] loBytes = bos.toByteArray();

			ByteArrayInputStream bis = new ByteArrayInputStream(loBytes);
			ObjectInput in = new ObjectInputStream(bis);
			loDeserialized = (ListObject) in.readObject();
		}
		catch(Exception e){
			System.out.println("Error while serializing and deserializing to bytes: " + e);
			assert(false);
		}

		MatrixObject mo1Deserialized = (MatrixObject) loDeserialized.getData(0);
		ListObject lotDeserialized = (ListObject) loDeserialized.getData(1);
		MatrixObject mo2Deserialized = (MatrixObject) lotDeserialized.getData(0);
		IntObject ioDeserialized = (IntObject) loDeserialized.getData(2);

		if (_named == 1)
			Assert.assertEquals(lo.getNames(), loDeserialized.getNames());

		Assert.assertArrayEquals(mo1.acquireRead().getDenseBlockValues(), mo1Deserialized.acquireRead().getDenseBlockValues(), 0);
		Assert.assertArrayEquals(mo2.acquireRead().getDenseBlockValues(), mo2Deserialized.acquireRead().getDenseBlockValues(), 0);
		Assert.assertEquals(io.getLongValue(), ioDeserialized.getLongValue());
	}

	@Test
	public void serializeListObjectProgramConverter() {
		MatrixObject mo1 = generateDummyMatrix(10);
		MatrixObject mo2 = generateDummyMatrix(20);
		IntObject io = new IntObject(30);
		ListObject lo;
		if (_named == 1)
			lo = new ListObject(Arrays.asList(mo1, mo2, io), Arrays.asList("e1", "e2", "e3"));
		else
			lo = new ListObject(Arrays.asList(mo1, mo2, io));

		String serial = ProgramConverter.serializeDataObject("key", lo);
		Object[] obj = ProgramConverter.parseDataObject(serial);
		ListObject actualLO = (ListObject) obj[1];
		MatrixObject actualMO1 = (MatrixObject) actualLO.slice(0);
		MatrixObject actualMO2 = (MatrixObject) actualLO.slice(1);
		IntObject actualIO = (IntObject) actualLO.slice(2);

		if (_named == 1)
			Assert.assertEquals(lo.getNames(), actualLO.getNames());

		Assert.assertArrayEquals(mo1.acquireRead().getDenseBlockValues(), actualMO1.acquireRead().getDenseBlockValues(), 0);
		Assert.assertArrayEquals(mo2.acquireRead().getDenseBlockValues(), actualMO2.acquireRead().getDenseBlockValues(), 0);
		Assert.assertEquals(io.getLongValue(), actualIO.getLongValue());
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
