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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.util.DataConverter;
import org.apache.sysml.runtime.util.UtilFunctions;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.utils.TestUtils;
import org.junit.Assert;
import org.junit.Test;

public class FrameIndexingTest extends AutomatedTestBase
{
	private final static int rows = 3345;
	private final static int rl = 234;
	private final static int ru = 1432;
	private final static int cl = 0;
	private final static int cu = 2;
	
	private final static ValueType[] schemaStrings = new ValueType[]{ValueType.STRING, ValueType.STRING, ValueType.STRING};	
	private final static ValueType[] schemaMixed = new ValueType[]{ValueType.STRING, ValueType.DOUBLE, ValueType.INT, ValueType.BOOLEAN};	
	
	private enum IXType {
		RIX,
		LIX,
	}
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
	}

	@Test
	public void testFrameStringsRIX()  {
		runFrameIndexingTest(schemaStrings, IXType.RIX);
	}
	
	@Test
	public void testFrameMixedRIX()  {
		runFrameIndexingTest(schemaMixed, IXType.RIX);
	}
	
	@Test
	public void testFrameStringsLIX()  {
		runFrameIndexingTest(schemaStrings, IXType.LIX);
	}
	
	@Test
	public void testFrameMixedLIX()  {
		runFrameIndexingTest(schemaMixed, IXType.LIX);
	}


	
	/**
	 * 
	 * @param sparseM1
	 * @param sparseM2
	 * @param instType
	 */
	private void runFrameIndexingTest( ValueType[] schema, IXType itype)
	{
		try
		{
			//data generation
			double[][] A = getRandomMatrix(rows, schema.length, -10, 10, 0.9, 2412); 
			
			//init data frame 1
			List<ValueType> lschema1 = Arrays.asList(schema);
			FrameBlock frame1 = new FrameBlock(lschema1);
			Object[] row1 = new Object[lschema1.size()];
			for( int i=0; i<rows; i++ ) {
				for( int j=0; j<lschema1.size(); j++ )
					A[i][j] = UtilFunctions.objectToDouble(lschema1.get(j), 
							row1[j] = UtilFunctions.doubleToObject(lschema1.get(j), A[i][j]));
				frame1.appendRow(row1);
			}
			
			//core indexing operation
			MatrixBlock mbC = null; 
			FrameBlock frame3 = null;
			if( itype == IXType.RIX ) 
			{
				//matrix indexing
				MatrixBlock mbA = DataConverter.convertToMatrixBlock(A);
				mbC = mbA.sliceOperations(rl, ru, cl, cu, new MatrixBlock());
				
				//frame indexing
				frame3 = frame1.sliceOperations(rl, ru, cl, cu, new FrameBlock());
			}
			else if( itype == IXType.LIX ) 
			{
				//data generation
				double[][] B = getRandomMatrix(ru-rl+1, cu-cl+1, -10, 10, 0.9, 7); 
				
				//init data frame 2
				List<ValueType> lschema2 = new ArrayList<ValueType>();
				for( int j=cl; j<=cu; j++ )
					lschema2.add(schema[j]);
				FrameBlock frame2 = new FrameBlock(lschema2);
				Object[] row2 = new Object[lschema2.size()];
				for( int i=0; i<ru-rl+1; i++ ) {
					for( int j=0; j<lschema2.size(); j++ )
						B[i][j] = UtilFunctions.objectToDouble(lschema2.get(j), 
								row2[j] = UtilFunctions.doubleToObject(lschema2.get(j), B[i][j]));
					frame2.appendRow(row2);
				}
				
				//matrix indexing
				MatrixBlock mbA = DataConverter.convertToMatrixBlock(A);
				MatrixBlock mbB = DataConverter.convertToMatrixBlock(B);
				mbC = mbA.leftIndexingOperations(mbB, rl, ru, cl, cu, new MatrixBlock(), false);
				
				//frame indexing
				frame3 = frame1.leftIndexingOperations(frame2, rl, ru, cl, cu, new FrameBlock());				
			}
			
			//check basic meta data
			if( frame3.getNumRows() != mbC.getNumRows() )
				Assert.fail("Wrong number of rows: "+frame3.getNumRows()+", expected: "+mbC.getNumRows());
		
			//check correct values
			List<ValueType> lschema = frame3.getSchema();
			for( int i=0; i<ru-rl+1; i++ ) 
				for( int j=0; j<lschema.size(); j++ )	{
					double tmp = UtilFunctions.objectToDouble(lschema.get(j), frame3.get(i, j));
					if( tmp != mbC.quickGetValue(i, j) )
						Assert.fail("Wrong get value for cell ("+i+","+j+"): "+tmp+", expected: "+mbC.quickGetValue(i, j));
				}		
		}
		catch(Exception ex) {
			ex.printStackTrace();
			throw new RuntimeException(ex);
		}
	}
}
