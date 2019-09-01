/*
 * Modifications Copyright 2019 Graz University of Technology
 *
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

package org.tugraz.sysds.test.component.frame;

import java.lang.reflect.Method;

import org.junit.Test;
import org.tugraz.sysds.common.Types.ValueType;
import org.tugraz.sysds.runtime.controlprogram.caching.CacheableData;
import org.tugraz.sysds.runtime.controlprogram.caching.FrameObject;
import org.tugraz.sysds.runtime.controlprogram.caching.LazyWriteBuffer;
import org.tugraz.sysds.runtime.matrix.data.FrameBlock;
import org.tugraz.sysds.runtime.matrix.data.InputInfo;
import org.tugraz.sysds.runtime.matrix.data.MatrixBlock;
import org.tugraz.sysds.runtime.matrix.data.OutputInfo;
import org.tugraz.sysds.runtime.meta.MatrixCharacteristics;
import org.tugraz.sysds.runtime.meta.MetaDataFormat;
import org.tugraz.sysds.runtime.util.DataConverter;
import org.tugraz.sysds.test.AutomatedTestBase;
import org.tugraz.sysds.test.TestUtils;

public class FrameEvictionTest extends AutomatedTestBase
{
	private final static int rows = 1593;
	private final static double sparsity1 = 0.9;
	private final static double sparsity2 = 0.1;		
	
	private final static ValueType[] schemaDoubles = new ValueType[]{ValueType.FP64, ValueType.FP64, ValueType.FP64};	
	private final static ValueType[] schemaStrings = new ValueType[]{ValueType.STRING, ValueType.STRING, ValueType.STRING};	
	private final static ValueType[] schemaMixed = new ValueType[]{ValueType.STRING, ValueType.FP64, ValueType.INT64, ValueType.BOOLEAN};	
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
	}

	@Test
	public void testFrameEvictionDoublesDenseDefault()  {
		runFrameEvictionTest(schemaDoubles, false, true, false);
	}
	
	@Test
	public void testFrameEvictionDoublesDenseCustom()  {
		runFrameEvictionTest(schemaDoubles, false, false, false);
	}
	
	@Test
	public void testFrameEvictionDoublesSparseDefault()  {
		runFrameEvictionTest(schemaDoubles, true, true, false);
	}
	
	@Test
	public void testFrameEvictionDoublesSparseCustom()  {
		runFrameEvictionTest(schemaDoubles, true, false, false);
	}
	
	@Test
	public void testFrameEvictionStringsDenseDefault()  {
		runFrameEvictionTest(schemaStrings, false, true, false);
	}
	
	@Test
	public void testFrameEvictionStringsDenseCustom()  {
		runFrameEvictionTest(schemaStrings, false, false, false);
	}
	
	@Test
	public void testFrameEvictionStringsSparseDefault()  {
		runFrameEvictionTest(schemaStrings, true, true, false);
	}
	
	@Test
	public void testFrameEvictionStringsSparseCustom()  {
		runFrameEvictionTest(schemaStrings, true, false, false);
	}
	
	@Test
	public void testFrameEvictionMixedDenseDefault()  {
		runFrameEvictionTest(schemaMixed, false, true, false);
	}
	
	@Test
	public void testFrameEvictionMixedDenseCustom()  {
		runFrameEvictionTest(schemaMixed, false, false, false);
	}
	
	@Test
	public void testFrameEvictionMixedSparseDefault()  {
		runFrameEvictionTest(schemaMixed, true, true, false);
	}
	
	@Test
	public void testFrameEvictionMixedSparseCustom()  {
		runFrameEvictionTest(schemaMixed, true, false, false);
	}

	@Test
	public void testFrameEvictionDoublesDenseDefaultForce()  {
		runFrameEvictionTest(schemaDoubles, false, true, true);
	}
	
	@Test
	public void testFrameEvictionDoublesDenseCustomForce()  {
		runFrameEvictionTest(schemaDoubles, false, false, true);
	}
	
	@Test
	public void testFrameEvictionDoublesSparseDefaultForce()  {
		runFrameEvictionTest(schemaDoubles, true, true, true);
	}
	
	@Test
	public void testFrameEvictionDoublesSparseCustomForce()  {
		runFrameEvictionTest(schemaDoubles, true, false, true);
	}
	
	@Test
	public void testFrameEvictionStringsDenseDefaultForce()  {
		runFrameEvictionTest(schemaStrings, false, true, true);
	}
	
	@Test
	public void testFrameEvictionStringsDenseCustomForce()  {
		runFrameEvictionTest(schemaStrings, false, false, true);
	}
	
	@Test
	public void testFrameEvictionStringsSparseDefaultForce()  {
		runFrameEvictionTest(schemaStrings, true, true, true);
	}
	
	@Test
	public void testFrameEvictionStringsSparseCustomForce()  {
		runFrameEvictionTest(schemaStrings, true, false, true);
	}
	
	@Test
	public void testFrameEvictionMixedDenseDefaultForce()  {
		runFrameEvictionTest(schemaMixed, false, true, true);
	}
	
	@Test
	public void testFrameEvictionMixedDenseCustomForce()  {
		runFrameEvictionTest(schemaMixed, false, false, true);
	}
	
	@Test
	public void testFrameEvictionMixedSparseDefaultForce()  {
		runFrameEvictionTest(schemaMixed, true, true, true);
	}
	
	@Test
	public void testFrameEvictionMixedSparseCustomForce()  {
		runFrameEvictionTest(schemaMixed, true, false, true);
	}

	
	/**
	 * 
	 * @param schema
	 * @param sparse
	 * @param defaultMeta
	 * @param force
	 */
	private void runFrameEvictionTest( ValueType[] schema, boolean sparse, boolean defaultMeta, boolean force)
	{
		try
		{
			//data generation
			double sparsity = sparse ? sparsity2 : sparsity1;
			double[][] A = getRandomMatrix(rows, schema.length, -10, 10, sparsity, 765); 
			MatrixBlock mA = DataConverter.convertToMatrixBlock(A);
			FrameBlock fA = DataConverter.convertToFrameBlock(mA, schema);
			
			//create non-default column names
			if( !defaultMeta ) {
				String[] colnames = new String[schema.length];
				for( int i=0; i<schema.length; i++ )
					colnames[i] = "Custom_name_"+i;
				fA.setColumnNames(colnames);
			}
		
			//setup caching
			CacheableData.initCaching("tmp_frame_eviction_test");
			
			//create frame object
			MatrixCharacteristics mc = new MatrixCharacteristics(rows, schema.length, -1, -1);
			MetaDataFormat meta = new MetaDataFormat (mc, 
					OutputInfo.BinaryBlockOutputInfo, InputInfo.BinaryBlockInputInfo);
			FrameObject fo = new FrameObject("fA",  meta, schema);
			fo.acquireModify(fA);
			fo.release();
			
			//evict frame and clear in-memory reference
			if( force )
				LazyWriteBuffer.forceEviction();
			Method clearfo = CacheableData.class
					.getDeclaredMethod("clearCache", new Class[]{});
			clearfo.setAccessible(true); //make method public
			clearfo.invoke(fo, new Object[]{});
			
			//read frame through buffer pool (if forced, this is a read from disk
			//otherwise deserialization or simple reference depending on schema)
			FrameBlock fA2 = fo.acquireRead();
			fo.release();
			
			//compare frames
			String[][] sA = DataConverter.convertToStringFrame(fA);
			String[][] sA2 = DataConverter.convertToStringFrame(fA2);
			TestUtils.compareFrames(sA, sA2, rows, schema.length);
		}
		catch(Exception ex) {
			ex.printStackTrace();
			throw new RuntimeException(ex);
		}
	}
}
