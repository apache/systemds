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

import java.util.Arrays;
import java.util.List;

import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.util.UtilFunctions;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.utils.TestUtils;
import org.junit.Assert;
import org.junit.Test;

public class FrameGetSetTest extends AutomatedTestBase
{
	private final static int rows = 3254;
	private final static ValueType[] schemaStrings = new ValueType[]{ValueType.STRING, ValueType.STRING, ValueType.STRING};	
	private final static ValueType[] schemaMixed = new ValueType[]{ValueType.STRING, ValueType.DOUBLE, ValueType.INT, ValueType.BOOLEAN};	
	
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
			List<ValueType> lschema = Arrays.asList(schema);
			FrameBlock frame = new FrameBlock(lschema);
			
			//init data frame 
			if( itype == InitType.COLUMN ) 
			{
				for( int j=0; j<lschema.size(); j++ ) {
					ValueType vt = lschema.get(j);
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
								A[i][j] = (tmp2[i] = (Boolean)UtilFunctions.doubleToObject(vt, A[i][j]))?1:0;
							frame.appendColumn(tmp2);
							break;
						case INT:
							long[] tmp3 = new long[rows];
							for( int i=0; i<rows; i++ )
								A[i][j] = tmp3[i] = (Long)UtilFunctions.doubleToObject(vt, A[i][j]);
							frame.appendColumn(tmp3);
							break;
						case DOUBLE:
							double[] tmp4 = new double[rows];
							for( int i=0; i<rows; i++ )
								tmp4[i] = (Double)UtilFunctions.doubleToObject(vt, A[i][j]);
							frame.appendColumn(tmp4);
							break;
						default:
							throw new RuntimeException("Unsupported value type: "+vt);
					}					
				}
			}
			else if( itype == InitType.ROW_OBJ ) {
				Object[] row = new Object[lschema.size()];
				for( int i=0; i<rows; i++ ) {
					for( int j=0; j<lschema.size(); j++ )
						A[i][j] = UtilFunctions.objectToDouble(lschema.get(j), 
								row[j] = UtilFunctions.doubleToObject(lschema.get(j), A[i][j]));
					frame.appendRow(row);
				}			
			}
			else if( itype == InitType.ROW_STRING ) {
				String[] row = new String[lschema.size()];
				for( int i=0; i<rows; i++ ) {
					for( int j=0; j<lschema.size(); j++ ) {
						Object obj = UtilFunctions.doubleToObject(lschema.get(j), A[i][j]);
						A[i][j] = UtilFunctions.objectToDouble(lschema.get(j), obj);
						row[j] = obj.toString();
					}
					frame.appendRow(row);
				}
			}
			
			//some updates via set
			for( int i=7; i<13; i++ )
				for( int j=0; j<=2; j++ ) {
					frame.set(i, j, UtilFunctions.doubleToObject(lschema.get(j), (double)i*j));
					A[i][j] = (double)i*j;
				}
			
			//check basic meta data
			if( frame.getNumRows() != rows )
				Assert.fail("Wrong number of rows: "+frame.getNumRows()+", expected: "+rows);
		
			//check correct values			
			for( int i=0; i<rows; i++ ) 
				for( int j=0; j<lschema.size(); j++ )	{
					double tmp = UtilFunctions.objectToDouble(lschema.get(j), frame.get(i, j));
					if( tmp != A[i][j] )
						Assert.fail("Wrong get value for cell ("+i+","+j+"): "+tmp+", expected: "+A[i][j]);
				}		
		}
		catch(Exception ex) {
			ex.printStackTrace();
			throw new RuntimeException(ex);
		}
	}
}
