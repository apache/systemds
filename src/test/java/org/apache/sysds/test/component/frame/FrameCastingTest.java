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
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestUtils;

public class FrameCastingTest extends AutomatedTestBase
{
	private final static int rows = 312;
	private final static ValueType[] schemaStrings = new ValueType[]{ValueType.STRING, ValueType.STRING, ValueType.STRING};	
	private final static ValueType[] schemaMixed = new ValueType[]{ValueType.STRING, ValueType.FP64, ValueType.INT64, ValueType.BITSET};
	
	private enum CastType {
		M2F_S,
		M2F_G,
		F2M,
	}
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
	}

	@Test
	public void testFrameStringsM2F_S() {
		runFrameCastingTest(schemaStrings, CastType.M2F_S);
	}
	
	@Test
	public void testFrameStringsM2F_G() {
		runFrameCastingTest(schemaStrings, CastType.M2F_G);
	}
	
	@Test
	public void testFrameStringsF2M() {
		runFrameCastingTest(schemaStrings, CastType.F2M);
	}
	
	@Test
	public void testFrameMixedM2F_S() {
		runFrameCastingTest(schemaMixed, CastType.M2F_S);
	}
	
	@Test
	public void testFrameMixedM2F_G() {
		runFrameCastingTest(schemaMixed, CastType.M2F_G);
	}
	
	@Test
	public void testFrameMixedF2M() {
		runFrameCastingTest(schemaMixed, CastType.F2M);
	}

	/**
	 * 
	 * @param sparseM1
	 * @param sparseM2
	 * @param instType
	 */
	private void runFrameCastingTest( ValueType[] schema, CastType ctype)
	{
		try
		{
			//data generation
			double[][] A = getRandomMatrix(rows, schema.length, -10, 10, 0.9, 2412); 
			for( int i=0; i<rows; i++ ) {
				for( int j=0; j<schema.length; j++ )
					A[i][j] = UtilFunctions.objectToDouble(schema[j], 
							  UtilFunctions.doubleToObject(schema[j], A[i][j]));
			}
			
			//core casting operations
			FrameBlock frame = null;
			if( ctype == CastType.F2M ) 
			{
				//construct input schema
				FrameBlock frame1 = new FrameBlock(schema);
				Object[] row1 = new Object[schema.length];
				for( int i=0; i<rows; i++ ) {
					for( int j=0; j<schema.length; j++ )
						row1[j] = UtilFunctions.doubleToObject(schema[j], A[i][j]);
					frame1.appendRow(row1);
				}

				MatrixBlock mb = DataConverter.convertToMatrixBlock(frame1);
				frame = DataConverter.convertToFrameBlock(mb);

			}
			else if( ctype == CastType.M2F_G )
			{
				MatrixBlock mb = DataConverter.convertToMatrixBlock(A);
				frame = DataConverter.convertToFrameBlock(mb);
			}
			else if( ctype == CastType.M2F_S )
			{
				MatrixBlock mb = DataConverter.convertToMatrixBlock(A);
				frame = DataConverter.convertToFrameBlock(mb, schema);	
			}
			
			if(frame != null){

				//check basic meta data
				if( frame.getNumRows() != rows )
					Assert.fail("Wrong number of rows: "+frame.getNumRows()+", expected: "+rows);
			
				//check correct values
				ValueType[] lschema = frame.getSchema();
				for( int i=0; i<rows; i++ ) {
					for( int j=0; j<lschema.length; j++ )	{
						double tmp = UtilFunctions.objectToDouble(lschema[j], frame.get(i, j));
						double tmpm = Double.isNaN(A[i][j]) ? 0.0: A[i][j];
						tmp = Double.isNaN(tmp) ? 0.0 : tmp;
	
						if( tmp != tmpm)
							Assert.fail("Wrong get value for cell ("+i+","+j+"): "+tmp+", expected: "+A[i][j]);
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
