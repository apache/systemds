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

package org.apache.sysds.test.functions.transform;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.transform.encode.ColumnEncoder;
import org.apache.sysds.runtime.transform.encode.ColumnEncoderBagOfWords;
import org.apache.sysds.runtime.transform.encode.ColumnEncoderComposite;
import org.apache.sysds.runtime.transform.encode.EncoderFactory;
import org.apache.sysds.runtime.transform.encode.MultiColumnEncoder;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;

public class ColumnEncoderSerializationTest extends AutomatedTestBase
{
	private final static int rows = 2791;
	private final static int cols = 8;

	private final static Types.ValueType[] schemaStrings = new Types.ValueType[]{
		Types.ValueType.STRING, Types.ValueType.STRING, Types.ValueType.STRING, Types.ValueType.STRING,
		Types.ValueType.STRING, Types.ValueType.STRING, Types.ValueType.STRING, Types.ValueType.STRING};
	private final static Types.ValueType[] schemaMixed = new Types.ValueType[]{
		Types.ValueType.STRING, Types.ValueType.FP64, Types.ValueType.INT64, Types.ValueType.BOOLEAN,
		Types.ValueType.STRING, Types.ValueType.FP64, Types.ValueType.INT64, Types.ValueType.BOOLEAN};


	public enum TransformType {
		RECODE,
		DUMMY,
		IMPUTE,
		OMIT,
		BOW
	}

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
	}

	@Test
	public void testComposite1() { runTransformSerTest(TransformType.DUMMY, schemaStrings); }

	@Test
	public void testComposite2() { runTransformSerTest(TransformType.RECODE, schemaMixed); }

	@Test
	public void testComposite3() { runTransformSerTest(TransformType.RECODE, schemaStrings); }

	@Test
	public void testComposite4() { runTransformSerTest(TransformType.DUMMY, schemaMixed); }

	@Test
	public void testComposite5() { runTransformSerTest(TransformType.IMPUTE, schemaMixed); }

	@Test
	public void testComposite6() { runTransformSerTest(TransformType.IMPUTE, schemaStrings); }

	@Test
	public void testComposite7() { runTransformSerTest(TransformType.OMIT, schemaMixed); }

	@Test
	public void testComposite8() { runTransformSerTest(TransformType.OMIT, schemaStrings); }

	@Test
	public void testComposite9() { runTransformSerTest(TransformType.BOW, schemaStrings); }

	@Test
	public void testComposite10() { runTransformSerTest(TransformType.BOW, schemaMixed); }




	private void runTransformSerTest(TransformType type, Types.ValueType[] schema) {
		//data generation
		double[][] A = getRandomMatrix(rows, cols, -10, 10, 0.9, 8234);

		//init data frame
		FrameBlock frame = new FrameBlock(schema);

		//init data frame
		Object[] row = new Object[schema.length];
		for( int i=0; i < rows; i++) {
			for( int j=0; j<schema.length; j++ )
				A[i][j] = UtilFunctions.objectToDouble(schema[j],
					row[j] = UtilFunctions.doubleToObject(schema[j], A[i][j]));
			frame.appendRow(row);
		}

		String spec = "";
		if(type == TransformType.DUMMY)
			spec = "{\n \"ids\": true\n, \"dummycode\":[ 2, 7, 8, 1 ]\n\n}";
		else if(type == TransformType.RECODE)
			spec = "{\n \"ids\": true\n, \"recode\":[ 2, 7, 1, 8 ]\n\n}";
		else if(type == TransformType.IMPUTE)
			spec = "{\n \"ids\": true\n, \"impute\":[ { \"id\": 6, \"method\": \"constant\", \"value\": \"1\" }, " +
					"{ \"id\": 7, \"method\": \"global_mode\" }, { \"id\": 9, \"method\": \"global_mean\" } ]\n\n}";
		else if (type == TransformType.OMIT)
			spec = "{ \"ids\": true, \"omit\": [ 1,2,4,5,6,7,8,9 ], \"recode\": [ 2, 7 ] }";
		else if (type == TransformType.BOW)
			spec = "{ \"ids\": true, \"omit\": [ 1,4,5,6,8,9 ], \"bag_of_words\": [ 2, 7 ] }";

		frame.setSchema(schema);
		String[] cnames = frame.getColumnNames();

		MultiColumnEncoder encoderIn = EncoderFactory.createEncoder(spec, cnames, frame.getNumColumns(), null);
		if(type == TransformType.BOW){
			List<ColumnEncoderBagOfWords> encs = encoderIn.getColumnEncoders(ColumnEncoderBagOfWords.class);
			HashMap<Object, Integer> dict = new HashMap<>();
			dict.put("val1", 1);
			dict.put("val2", 2);
			dict.put("val3", 300);
			encs.forEach(e -> e.setTokenDictionary(dict));
		}
		MultiColumnEncoder encoderOut;

		// serialization and deserialization
		encoderOut = serializeDeserialize(encoderIn);
		// compare
		assert encoderOut != null;
		Assert.assertArrayEquals(encoderIn.getFromAllIntArray(ColumnEncoderComposite.class, ColumnEncoder::getColID),
				encoderOut.getFromAllIntArray(ColumnEncoderComposite.class, ColumnEncoder::getColID));

		int numIn = encoderIn.getColumnEncoders().size();
		int numOut = encoderOut.getColumnEncoders().size();
		Assert.assertEquals(numIn, numOut);
		List<Class<? extends ColumnEncoder>> typesIn = encoderIn.getEncoderTypes();
		List<Class<? extends ColumnEncoder>> typesOut = encoderOut.getEncoderTypes();
		Assert.assertArrayEquals(typesIn.toArray(), typesOut.toArray());

		for(Class<? extends ColumnEncoder> classtype: typesIn){
			Assert.assertArrayEquals(encoderIn.getFromAllIntArray(classtype, ColumnEncoder::getColID), encoderOut.getFromAllIntArray(classtype, ColumnEncoder::getColID));
		}
		if(type == TransformType.BOW){
			List<ColumnEncoderBagOfWords> encsIn = encoderIn.getColumnEncoders(ColumnEncoderBagOfWords.class);
			List<ColumnEncoderBagOfWords> encsOut = encoderOut.getColumnEncoders(ColumnEncoderBagOfWords.class);
			for (int i = 0; i < encsIn.size(); i++) {
				Map<Object, Integer> eOutDict = encsOut.get(i).getTokenDictionary();
				encsIn.get(i).getTokenDictionary().forEach((k,v) -> {
					assert v.equals(eOutDict.get(k));
				});
			}
		}

	}

	private static MultiColumnEncoder serializeDeserialize(MultiColumnEncoder encoderIn) {
		try {
			ByteArrayOutputStream bos = new ByteArrayOutputStream();
			ObjectOutputStream oos = new ObjectOutputStream(bos);
			oos.writeObject(encoderIn);
			oos.flush();
			byte[] encoderBytes = bos.toByteArray();

			ByteArrayInputStream bis = new ByteArrayInputStream(encoderBytes);
			ObjectInput in = new ObjectInputStream(bis);
			return (MultiColumnEncoder) in.readObject();
		}
		catch(IOException | ClassNotFoundException e) {
			e.printStackTrace();
		}
		return null;
	}
}