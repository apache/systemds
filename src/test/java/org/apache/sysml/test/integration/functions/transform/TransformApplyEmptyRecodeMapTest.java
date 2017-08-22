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

package org.apache.sysml.test.integration.functions.transform;

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.transform.encode.Encoder;
import org.apache.sysml.runtime.transform.encode.EncoderFactory;
import org.apache.sysml.runtime.util.DataConverter;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.utils.TestUtils;

public class TransformApplyEmptyRecodeMapTest extends AutomatedTestBase 
{
	private static final int rows = 7;
	private static final int cols = 1;
	
	@Override
	public void setUp()  {
		TestUtils.clearAssertionInformation();
	}
	
	@Test
	public void testTransformApplyEmptyRecodeMap() {
		try {
			//generate input data
			FrameBlock data = DataConverter.convertToFrameBlock(
				DataConverter.convertToMatrixBlock(getRandomMatrix(rows, cols, 1, 1, 1, 7)));
			FrameBlock meta = new FrameBlock(new ValueType[]{ValueType.STRING}, new String[]{"C1"});
			
			//execute transform apply
			Encoder encoder = EncoderFactory.createEncoder(
				"{ids:true, recode:[1]}", data.getColumnNames(), meta.getSchema(), meta);
			MatrixBlock out = encoder.apply(data, new MatrixBlock(rows, cols, true));
			
			//check outputs
			Assert.assertEquals(rows, out.getNumRows());
			Assert.assertEquals(cols, out.getNumColumns());
			for(int i=0; i<rows; i++)
				for(int j=0; j<cols; j++)
					Assert.assertTrue(Double.isNaN(out.quickGetValue(i, j)));
		} 
		catch (DMLRuntimeException e) {
			throw new RuntimeException(e);
		}
	}
}
