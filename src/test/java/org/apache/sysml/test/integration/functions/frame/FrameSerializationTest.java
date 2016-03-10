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

package org.apache.sysml.test.integration.functions.frame;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Arrays;
import java.util.List;

import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.util.UtilFunctions;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.utils.TestUtils;
import org.junit.Assert;
import org.junit.Test;

public class FrameSerializationTest extends AutomatedTestBase
{
	private final static int rows = 2791;
	private final static ValueType[] schemaStrings = new ValueType[]{ValueType.STRING, ValueType.STRING, ValueType.STRING};	
	private final static ValueType[] schemaMixed = new ValueType[]{ValueType.STRING, ValueType.DOUBLE, ValueType.INT, ValueType.BOOLEAN};	
	
	private enum SerType {
		WRITABLE_SER,
		JAVA_SER,
	}
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
	}

	@Test
	public void testFrameStringsWritable()  {
		runFrameSerializeTest(schemaStrings, SerType.WRITABLE_SER);
	}
	
	@Test
	public void testFrameMixedWritable()  {
		runFrameSerializeTest(schemaMixed, SerType.WRITABLE_SER);
	}
	
	@Test
	public void testFrameStringsJava()  {
		runFrameSerializeTest(schemaStrings, SerType.JAVA_SER);
	}
	
	@Test
	public void testFrameMixedJava()  {
		runFrameSerializeTest(schemaMixed, SerType.JAVA_SER);
	}

	
	/**
	 * 
	 * @param sparseM1
	 * @param sparseM2
	 * @param instType
	 */
	private void runFrameSerializeTest( ValueType[] schema, SerType stype)
	{
		try
		{
			//data generation
			double[][] A = getRandomMatrix(rows, schema.length, -10, 10, 0.9, 8234); 
			
			//init data frame
			List<ValueType> lschema = Arrays.asList(schema);
			FrameBlock frame = new FrameBlock(lschema);
			
			//init data frame 
			Object[] row = new Object[lschema.size()];
			for( int i=0; i<rows; i++ ) {
				for( int j=0; j<lschema.size(); j++ )
					A[i][j] = fromObject(lschema.get(j), row[j] = toObject(lschema.get(j), A[i][j]));
				frame.appendRow(row);
			}			
			
			//core serialization and deserialization
			if( stype == SerType.WRITABLE_SER ) {
				//serialization
				ByteArrayOutputStream bos = new ByteArrayOutputStream();
				DataOutputStream dos = new DataOutputStream(bos);
				frame.write(dos);
				
				//deserialization
				ByteArrayInputStream bis = new ByteArrayInputStream(bos.toByteArray());
				DataInputStream dis = new DataInputStream(bis);
				frame = new FrameBlock();
				frame.readFields(dis);
			}
			else if( stype == SerType.JAVA_SER ) {
				//serialization
				ByteArrayOutputStream bos = new ByteArrayOutputStream();
				ObjectOutputStream oos = new ObjectOutputStream(bos);
				oos.writeObject(frame);
				
				//deserialization
				ByteArrayInputStream bis = new ByteArrayInputStream(bos.toByteArray());
				ObjectInputStream ois = new ObjectInputStream(bis);
				frame = (FrameBlock) ois.readObject();
			}
			
			//check basic meta data
			if( frame.getNumRows() != rows )
				Assert.fail("Wrong number of rows: "+frame.getNumRows()+", expected: "+rows);
		
			//check correct values			
			for( int i=0; i<rows; i++ ) 
				for( int j=0; j<lschema.size(); j++ )	{
					double tmp = fromObject(lschema.get(j), frame.get(i, j));
					if( tmp != A[i][j] )
						Assert.fail("Wrong get value for cell ("+i+","+j+"): "+tmp+", expected: "+A[i][j]);
				}		
		}
		catch(Exception ex) {
			ex.printStackTrace();
			throw new RuntimeException(ex);
		}
	}
	
	/**
	 * 
	 * @param vt
	 * @param in
	 * @return
	 */
	private Object toObject(ValueType vt, double in) {
		switch( vt ) {
			case STRING: return String.valueOf(in);
			case BOOLEAN: return (in!=0);
			case INT: return UtilFunctions.toLong(in);
			case DOUBLE: return in;
			default: throw new RuntimeException("Unsupported value type: "+vt);
		}
	}
	
	/**
	 * 
	 * @param vt
	 * @param in
	 * @return
	 */
	private double fromObject(ValueType vt, Object in) {
		switch( vt ) {
			case STRING: return Double.parseDouble((String)in);
			case BOOLEAN: return ((Boolean)in)?1d:0d;
			case INT: return (Long)in;
			case DOUBLE: return (Double)in;
			default: throw new RuntimeException("Unsupported value type: "+vt);
		}
	}
}
