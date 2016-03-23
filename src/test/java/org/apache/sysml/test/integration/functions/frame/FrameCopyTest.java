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
import org.apache.sysml.runtime.instructions.cp.AppendCPInstruction.AppendType;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.util.UtilFunctions;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.utils.TestUtils;
import org.junit.Assert;
import org.junit.Test;

public class FrameCopyTest extends AutomatedTestBase
{
	private final static int rows = 1593;
	private final static ValueType[] schemaStrings = new ValueType[]{ValueType.STRING, ValueType.STRING, ValueType.STRING};	
	private final static ValueType[] schemaMixed = new ValueType[]{ValueType.STRING, ValueType.DOUBLE, ValueType.INT, ValueType.BOOLEAN};	
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
	}

	@Test
	public void testFrameStringsStringsCBind()  {
		runFrameCopyTest(schemaStrings, schemaStrings, AppendType.CBIND);
	}
	
	@Test
	public void testFrameStringsStringsRBind()  { //note: ncol(A)=ncol(B)
		runFrameCopyTest(schemaStrings, schemaStrings, AppendType.RBIND);
	}
	
	@Test
	public void testFrameMixedStringsCBind()  {
		runFrameCopyTest(schemaMixed, schemaStrings, AppendType.CBIND);
	}
	
	@Test
	public void testFrameStringsMixedCBind()  {
		runFrameCopyTest(schemaStrings, schemaMixed, AppendType.CBIND);
	}
	
	@Test
	public void testFrameMixedMixedCBind()  {
		runFrameCopyTest(schemaMixed, schemaMixed, AppendType.CBIND);
	}
	
	@Test
	public void testFrameMixedMixedRBind()  { //note: ncol(A)=ncol(B)
		runFrameCopyTest(schemaMixed, schemaMixed, AppendType.RBIND);
	}

	
	/**
	 * 
	 * @param sparseM1
	 * @param sparseM2
	 * @param instType
	 */
	private void runFrameCopyTest( ValueType[] schema1, ValueType[] schema2, AppendType atype)
	{
		try
		{
			//data generation
			double[][] A = getRandomMatrix(rows, schema1.length, -10, 10, 0.9, 2373); 
			double[][] B = getRandomMatrix(rows, schema2.length, -10, 10, 0.9, 129); 
			
			//init data frame 1
			List<ValueType> lschema1 = Arrays.asList(schema1);
			FrameBlock frame1 = new FrameBlock(lschema1);
			Object[] row1 = new Object[lschema1.size()];
			for( int i=0; i<rows; i++ ) {
				for( int j=0; j<lschema1.size(); j++ )
					A[i][j] = UtilFunctions.objectToDouble(lschema1.get(j), 
							row1[j] = UtilFunctions.doubleToObject(lschema1.get(j), A[i][j]));
				frame1.appendRow(row1);
			}
			
			//init data frame 2
			List<ValueType> lschema2 = Arrays.asList(schema2);
			FrameBlock frame2 = new FrameBlock(lschema2);
			Object[] row2 = new Object[lschema2.size()];
			for( int i=0; i<rows; i++ ) {
				for( int j=0; j<lschema2.size(); j++ )
					B[i][j] = UtilFunctions.objectToDouble(lschema2.get(j), 
							row2[j] = UtilFunctions.doubleToObject(lschema2.get(j), B[i][j]));
				frame2.appendRow(row2);
			}
			
			//copy from one frame to another.
			FrameBlock frame1Backup = new FrameBlock(frame1.getSchema(), frame1.getColumnNames());
			frame1Backup.copy(frame1);
			
			FrameBlock frame2Backup = new FrameBlock(frame2.getSchema(), frame2.getColumnNames());
			frame2Backup.copy(frame2);
			
			// Verify copied data.
			List<ValueType> lschema = frame1.getSchema();
			for ( int i=0; i<frame1.getNumRows(); ++i )
				for( int j=0; j<lschema.size(); j++ )	{
					if( UtilFunctions.compareTo(lschema.get(j), frame1.get(i, j), frame1Backup.get(i, j)) != 0)
						Assert.fail("Target value for cell ("+ i + "," + j + ") is " + frame1.get(i,  j) + 
								", is not same as original value " + frame1Backup.get(i, j));
				}		
			
			lschema = frame2.getSchema();
			for ( int i=0; i<frame2.getNumRows(); ++i )
				for( int j=0; j<lschema.size(); j++ )	{
					if( UtilFunctions.compareTo(lschema.get(j), frame2.get(i, j), frame2Backup.get(i, j)) != 0)
						Assert.fail("Target value for cell ("+ i + "," + j + ") is " + frame2.get(i,  j) + 
								", is not same as original value " + frame2Backup.get(i, j));
				}		

			// update some data in original/backup frames
			int updateRow = rows/2;
			lschema = frame1.getSchema();
			for( int j=0; j<lschema.size(); j++ )	{
				switch( lschema.get(j) ) {
					case STRING:  frame1.set(updateRow,  j,  "String:"+ frame1.get(updateRow, j)); break;
					case BOOLEAN: frame1.set(updateRow,  j, ((Boolean)frame1.get(updateRow, j))?(new Boolean(false)):(new Boolean(true))); break;
					case INT:     frame1.set(updateRow,  j, (Long)frame1.get(updateRow, j) * 2 + 5); break;
					case DOUBLE:  frame1.set(updateRow,  j, (Double)frame1.get(updateRow, j) * 2 + 7); break;
					default: throw new RuntimeException("Unsupported value type: "+lschema.get(j));
				}
			}		
			
			lschema = frame2.getSchema();
			for( int j=0; j<lschema.size(); j++ )	{
				switch( lschema.get(j) ) {
					case STRING:  frame2.set(updateRow,  j,  "String:"+ frame2.get(updateRow, j)); break;
					case BOOLEAN: frame2.set(updateRow,  j, ((Boolean)frame2.get(updateRow, j))?(new Boolean(false)):(new Boolean(true))); break;
					case INT:     frame2.set(updateRow,  j, (Long)frame2.get(updateRow, j) * 2 + 4); break;
					case DOUBLE:  frame2.set(updateRow,  j, (Double)frame2.get(updateRow, j) * 2 + 6); break;
					default: throw new RuntimeException("Unsupported value type: "+lschema.get(j));
				}
			}		

			// Verify that data modified only on target frames
			lschema = frame1.getSchema();
			for( int j=0; j<lschema.size(); j++ )	{
				if( UtilFunctions.compareTo(lschema.get(j), frame1.get(updateRow, j), frame1Backup.get(updateRow, j)) == 0)
					Assert.fail("Updated value for cell ("+ updateRow + "," + j + ") is " + frame1.get(updateRow,  j) + 
							", same as original value "+frame1Backup.get(updateRow, j));
			}		
			
			lschema = frame2.getSchema();
			for( int j=0; j<lschema.size(); j++ )	{
				if( UtilFunctions.compareTo(lschema.get(j), frame2.get(updateRow, j), frame2Backup.get(updateRow, j)) == 0)
					Assert.fail("Updated value for cell ("+ updateRow + "," + j + ") is " + frame2.get(updateRow,  j) + 
							", same as original value "+frame2Backup.get(updateRow, j));
			}		
			
		}
		catch(Exception ex) {
			ex.printStackTrace();
			throw new RuntimeException(ex);
		}
	}
}
