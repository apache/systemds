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

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject.UpdateType;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestUtils;

public class FrameIndexingTest extends AutomatedTestBase
{
	private final static int rows = 100;
	private final static int rl = 13;
	private final static int ru = 55;
	private final static int cl = 0;
	private final static int cu = 2;
	
	private final static ValueType[] schemaStrings = new ValueType[]{ValueType.STRING, ValueType.STRING, ValueType.STRING};	
	private final static ValueType[] schemaMixed = new ValueType[]{ValueType.STRING, ValueType.FP64, ValueType.INT64, ValueType.BOOLEAN};
	
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
			FrameBlock frame1 = new FrameBlock(schema);
			Object[] row1 = new Object[schema.length];
			for( int i=0; i<rows; i++ ) {
				for( int j=0; j<schema.length; j++ )
					A[i][j] = UtilFunctions.objectToDouble(schema[j], 
							row1[j] = UtilFunctions.doubleToObject(schema[j], A[i][j]));
				frame1.appendRow(row1);
			}
			
			//core indexing operation
			MatrixBlock mbC = null; 
			FrameBlock frame3 = null;
			if( itype == IXType.RIX ) 
			{
				//matrix indexing
				MatrixBlock mbA = DataConverter.convertToMatrixBlock(A);
				mbC = mbA.slice(rl, ru, cl, cu, new MatrixBlock());
				
				//frame indexing
				frame3 = frame1.slice(rl, ru, cl, cu, new FrameBlock());
			}
			else if( itype == IXType.LIX ) 
			{
				//data generation
				double[][] B = getRandomMatrix(ru-rl+1, cu-cl+1, -10, 10, 0.9, 7); 
				
				//init data frame 2
				ValueType[] lschema2 = new ValueType[cu-cl+1];
				for( int j=cl; j<=cu; j++ )
					lschema2[j-cl] = schema[j];
				FrameBlock frame2 = new FrameBlock(lschema2);
				Object[] row2 = new Object[lschema2.length];
				for( int i=0; i<ru-rl+1; i++ ) {
					for( int j=0; j<lschema2.length; j++ )
						B[i][j] = UtilFunctions.objectToDouble(lschema2[j], 
								row2[j] = UtilFunctions.doubleToObject(lschema2[j], B[i][j]));
					frame2.appendRow(row2);
				}
				
				//matrix indexing
				MatrixBlock mbA = DataConverter.convertToMatrixBlock(A);
				MatrixBlock mbB = DataConverter.convertToMatrixBlock(B);
				mbC = mbA.leftIndexingOperations(mbB, rl, ru, cl, cu, new MatrixBlock(), UpdateType.COPY);
				
				//frame indexing
				frame3 = frame1.leftIndexingOperations(frame2, rl, ru, cl, cu, new FrameBlock());				
			}
			
			//check basic meta data
			if(frame3 != null && mbC != null &&  frame3.getNumRows() != mbC.getNumRows() )
				Assert.fail("Wrong number of rows: "+frame3.getNumRows()+", expected: "+mbC.getNumRows());
		
			//check correct values
			if(frame3 != null && mbC != null){
				ValueType[] lschema = frame3.getSchema();
				for( int i=0; i<ru-rl+1; i++ ) {
					for( int j=0; j<lschema.length; j++ )	{
						double tmp = UtilFunctions.objectToDouble(lschema[j], frame3.get(i, j));
						if( tmp != mbC.quickGetValue(i, j) )
							Assert.fail("Wrong get value for cell ("+i+","+j+"): "+tmp+", expected: "+mbC.quickGetValue(i, j));
					}		
				}
			}
		}
		catch(Exception ex) {
			ex.printStackTrace();
			throw new RuntimeException(ex);
		}
	}
}
