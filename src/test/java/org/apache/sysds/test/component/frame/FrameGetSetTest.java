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

import org.apache.sysds.runtime.frame.data.columns.Array;
import org.apache.sysds.runtime.frame.data.columns.ArrayFactory;
import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestUtils;

public class FrameGetSetTest extends AutomatedTestBase
{
	private final static int rows = 131;
	private final static ValueType[] schemaStrings = new ValueType[]{ValueType.STRING, ValueType.STRING, ValueType.STRING};	
	private final static ValueType[] schemaMixed = new ValueType[]{ValueType.STRING, ValueType.FP64, ValueType.INT64, ValueType.BOOLEAN};	
	
	private enum InitType {
		COLUMN,
		ROW_OBJ,
		ROW_STRING,
	}
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
	}

	@Test
	public void testFrameStringsColumn()  {
		runFrameGetSetTest(schemaStrings, InitType.COLUMN);
	}
	
	@Test
	public void testFrameMixedColumn()  {
		runFrameGetSetTest(schemaMixed, InitType.COLUMN);
	}
	
	@Test
	public void testFrameStringsRowObj()  {
		runFrameGetSetTest(schemaStrings, InitType.ROW_OBJ);
	}
	
	@Test
	public void testFrameMixedRowObj()  {
		runFrameGetSetTest(schemaMixed, InitType.ROW_OBJ);
	}
	
	@Test
	public void testFrameStringsRowString()  {
		runFrameGetSetTest(schemaStrings, InitType.ROW_STRING);
	}
	
	@Test
	public void testFrameMixedRowString()  {
		runFrameGetSetTest(schemaMixed, InitType.ROW_STRING);
	}

	
	/**
	 * 
	 * @param sparseM1
	 * @param sparseM2
	 * @param instType
	 */
	private void runFrameGetSetTest( ValueType[] schema, InitType itype)
	{
		try
		{
			//data generation
			double[][] A = getRandomMatrix(rows, schema.length, -10, 10, 0.9, 8234); 
			
			//init data frame
			FrameBlock frame = new FrameBlock(schema);
			
			//init data frame 
			if( itype == InitType.COLUMN ) 
			{
				for( int j=0; j<schema.length; j++ ) {
					ValueType vt = schema[j];
					switch( vt ) {
						case STRING: 
							String[] tmp1 = new String[rows];
							for( int i=0; i<rows; i++ )
								tmp1[i] = (String)UtilFunctions.doubleToObject(vt, A[i][j]);
							frame.appendColumn(tmp1);
							break;
						case BOOLEAN:
							boolean[] tmp2 = new boolean[rows];
							for( int i=0; i<rows; i++ )
								A[i][j] = (tmp2[i] = (Boolean)UtilFunctions.doubleToObject(vt, A[i][j], false))?1:0;
							frame.appendColumn(tmp2);
							break;
						case INT64:
							long[] tmp3 = new long[rows];
							for( int i=0; i<rows; i++ )
								A[i][j] = tmp3[i] = (Long)UtilFunctions.doubleToObject(vt, A[i][j], false);
							frame.appendColumn(tmp3);
							break;
						case FP64:
							double[] tmp4 = new double[rows];
							for( int i=0; i<rows; i++ )
								tmp4[i] = (Double)UtilFunctions.doubleToObject(vt, A[i][j], false);
							frame.appendColumn(tmp4);
							break;
						default:
							throw new RuntimeException("Unsupported value type: "+vt);
					}					
				}
			}
			else if( itype == InitType.ROW_OBJ ) {
				Object[] row = new Object[schema.length];
				for( int i=0; i<rows; i++ ) {
					for( int j=0; j<schema.length; j++ )
						A[i][j] = UtilFunctions.objectToDouble(schema[j], 
								row[j] = UtilFunctions.doubleToObject(schema[j], A[i][j]));
					frame.appendRow(row);
				}			
			}
			else if( itype == InitType.ROW_STRING ) {
				String[] row = new String[schema.length];
				for( int i=0; i<rows; i++ ) {
					for( int j=0; j<schema.length; j++ ) {
						Object obj = UtilFunctions.doubleToObject(schema[j], A[i][j]);
						A[i][j] = UtilFunctions.objectToDouble(schema[j], obj);
						row[j] = (obj!=null) ? obj.toString() : null;
					}
					frame.appendRow(row);
				}
			}
			
			//some updates via set
			for( int i=7; i<13; i++ )
				for( int j=0; j<=2; j++ ) {
					frame.set(i, j, UtilFunctions.doubleToObject(schema[j], (double)i*j));
					A[i][j] = (double)i*j;
				}
			
			//check basic meta data
			if( frame.getNumRows() != rows )
				Assert.fail("Wrong number of rows: "+frame.getNumRows()+", expected: "+rows);
		
			//check correct values			
			for( int i=0; i<rows; i++ ) 
				for( int j=0; j<schema.length; j++ )	{
					double tmp = UtilFunctions.objectToDouble(schema[j], frame.get(i, j));
					if( tmp != A[i][j] )
						Assert.fail("Wrong get value for cell ("+i+","+j+"): "+tmp+", expected: "+A[i][j]);
				}		
		}
		catch(Exception ex) {
			ex.printStackTrace();
			throw new RuntimeException(ex);
		}
	}


	@Test
	public void testSetRow() {
		FrameBlock frame = new FrameBlock(schemaMixed, "0", rows);

		frame.setRow(2, new Object[] {"2", 2.0, 2L, true});

		assertEquals(frame.get(2, 0), "2");
		assertEquals(frame.get(2, 1), 2.0);
		assertEquals(frame.get(2, 2), 2L);
		assertEquals(frame.get(2, 3), true);
	}

	@Test
	public void testAppendColumnChunk() {
		FrameBlock frame = new FrameBlock(schemaMixed, rows);

		Array<?> chunk = ArrayFactory.create(new double[] {1.0, 2.0});
		Array<?> chunk2 = ArrayFactory.create(new double[] {3.0, 4.0});
		frame.appendColumnChunk(1, chunk);
		frame.appendColumnChunk(1, chunk2);

		assertEquals(frame.get(0, 1), 1.0);
		assertEquals(frame.get(1, 1), 2.0);
		assertEquals(frame.get(2, 1), 3.0);
		assertEquals(frame.get(3, 1), 4.0);
	}

	@Test
	public void testSetColumnChunk() {
		FrameBlock frame = new FrameBlock(schemaMixed, "0", rows);

		Array<?> chunk = ArrayFactory.create(new double[] {1.0, 2.0});
		frame.setColumnChunk(1, chunk, 5, rows);

		assertEquals(frame.get(5, 1), 1.0);
		assertEquals(frame.get(6, 1), 2.0);
	}
}
