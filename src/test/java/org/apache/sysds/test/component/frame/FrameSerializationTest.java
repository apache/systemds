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

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Collection;

import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.test.component.frame.array.FrameArrayTests;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public class FrameSerializationTest {

	private enum SerType {
		WRITABLE_SER, JAVA_SER,
	}

	private final FrameBlock frame;
	private final SerType type;

	@Parameters
	public static Collection<Object[]> data() {
		ArrayList<Object[]> tests = new ArrayList<>();

		ValueType[] schemaStrings = new ValueType[] {ValueType.STRING, ValueType.STRING, ValueType.STRING};
		ValueType[] schemaMixed = new ValueType[] {ValueType.STRING, ValueType.FP64, ValueType.INT64, ValueType.BITSET};
		ValueType[] rand = TestUtils.generateRandomSchema(10, 3);
		ValueType[] rand2 = TestUtils.generateRandomSchema(10, 4);
		try {
			for(ValueType[] sch : new ValueType[][] {schemaStrings, schemaMixed, rand, rand2}) {
				for(SerType t : SerType.values()) {
					tests.add(new Object[] {TestUtils.generateRandomFrameBlock(10, sch, 32), t});
					tests.add(new Object[] {new FrameBlock(sch), t});
					tests.add(new Object[] {new FrameBlock(sch, FrameArrayTests.generateRandomString(sch.length, 32)), t});
				}
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("failed constructing tests");
		}

		return tests;
	}

	public FrameSerializationTest(FrameBlock frame, SerType type) {
		this.frame = frame;
		this.type = type;
	}

	@Test
	public void serializeTest() {
		try {
			// init data frame
			FrameBlock back;
			// core serialization and deserialization
			if(type == SerType.WRITABLE_SER)
				back = writableSerialize(frame);
			else // if(stype == SerType.JAVA_SER)
				back = javaSerialize(frame);
			TestUtils.compareFrames(frame, back, true);
		}
		catch(Exception ex) {
			ex.printStackTrace();
			throw new RuntimeException("Failed serializing : " + frame.toString(), ex);
		}
	}

	@Test
	public void serializeReuse() {
		// verify that the serialization into already allocated Frame reuse the arrays.
		try {
			FrameBlock back = new FrameBlock();
			back = writableSerialize(frame, back);
			ValueType[] v1 = back.getSchema();
			back = writableSerialize(frame, back);
			ValueType[] v2 = back.getSchema();
			assertTrue(v1 == v2); // object equivalence !
			TestUtils.compareFrames(frame, back, true);
		}
		catch(Exception ex) {
			ex.printStackTrace();
			throw new RuntimeException("Failed serializing : " + frame.toString(), ex);
		}
	}

	@Test
	public void serializeDonNotReuseIfDifferentColumns() {
		// verify that the serialization into already allocated Frame reuse the arrays.
		try {
			if(frame.getNumColumns() == 1 || frame.getNumRows() < 10)
				return; // not valid test
			FrameBlock back = new FrameBlock();
			back = writableSerialize(frame, back);
			back = back.slice(0, frame.getNumRows()-1 , 0, 0);
			ValueType[] v1 = back.getSchema();
			back = writableSerialize(frame, back);
			ValueType[] v2 = back.getSchema();
			assertFalse(v1 == v2); // object equivalence !
			TestUtils.compareFrames(frame, back, true);
		}
		catch(Exception ex) {
			ex.printStackTrace();
			throw new RuntimeException("Failed serializing : " + frame.toString(), ex);
		}
	}

	@Test
	public void estimateMemory() {
		// should always be true that in memory size is bigger than serialized size.
		assertTrue(frame.getInMemorySize() > frame.getExactSerializedSize());
	}

	private static FrameBlock writableSerialize(FrameBlock in) throws Exception {
		return writableSerialize(in, new FrameBlock());
	}

	private static FrameBlock writableSerialize(FrameBlock in, FrameBlock ret) throws Exception {
		// serialization
		ByteArrayOutputStream bos = new ByteArrayOutputStream();
		DataOutputStream dos = new DataOutputStream(bos);
		in.write(dos);

		// deserialization
		ByteArrayInputStream bis = new ByteArrayInputStream(bos.toByteArray());
		DataInputStream dis = new DataInputStream(bis);
		ret.readFields(dis);
		return ret;
	}

	private static FrameBlock javaSerialize(FrameBlock in) throws Exception {
		// serialization
		ByteArrayOutputStream bos = new ByteArrayOutputStream();
		ObjectOutputStream oos = new ObjectOutputStream(bos);
		oos.writeObject(in);

		// deserialization
		ByteArrayInputStream bis = new ByteArrayInputStream(bos.toByteArray());
		ObjectInputStream ois = new ObjectInputStream(bis);
		return (FrameBlock) ois.readObject();
	}
}
