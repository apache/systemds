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
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.util.DataConverter;
import org.apache.sysml.runtime.util.UtilFunctions;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.utils.TestUtils;
import org.junit.Assert;
import org.junit.Test;

public class FrameCastingTest extends AutomatedTestBase
{
	private final static int rows = 2891;
	private final static ValueType[] schemaStrings = new ValueType[]{ValueType.STRING, ValueType.STRING, ValueType.STRING};	
	private final static ValueType[] schemaMixed = new ValueType[]{ValueType.STRING, ValueType.DOUBLE, ValueType.INT, ValueType.BOOLEAN};	
	
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
				List<ValueType> lschema1 = Arrays.asList(schema);
				FrameBlock frame1 = new FrameBlock(lschema1);
				Object[] row1 = new Object[lschema1.size()];
				for( int i=0; i<rows; i++ ) {
					for( int j=0; j<lschema1.size(); j++ )
						row1[j] = UtilFunctions.doubleToObject(lschema1.get(j), A[i][j]);
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
				frame = DataConverter.convertToFrameBlock(mb, Arrays.asList(schema));	
			}
			
			//check basic meta data
			if( frame.getNumRows() != rows )
				Assert.fail("Wrong number of rows: "+frame.getNumRows()+", expected: "+rows);
		
			//check correct values
			List<ValueType> lschema = frame.getSchema();
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
