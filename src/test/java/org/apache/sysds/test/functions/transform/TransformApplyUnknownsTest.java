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

package org.apache.sysds.test.functions.transform;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.transform.encode.EncoderFactory;
import org.apache.sysds.runtime.transform.encode.MultiColumnEncoder;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;

public class TransformApplyUnknownsTest extends AutomatedTestBase {
	protected static final Log LOG = LogFactory.getLog(TransformApplyUnknownsTest.class.getName());
	private static final int rows = 70;
	
	@Override
	public void setUp()  {
		TestUtils.clearAssertionInformation();
	}
	
	@Test
	public void testTransformApplyRecode() {
		try {
			//generate input data
			FrameBlock data = DataConverter.convertToFrameBlock(MatrixBlock.seqOperations(1, rows, 1));
			FrameBlock data2 = DataConverter.convertToFrameBlock(MatrixBlock.seqOperations(1, rows+10, 1));
			
			//encode and obtain meta data
			String spec = "{ids:true, recode:[1]}";
			MultiColumnEncoder encoder = EncoderFactory.createEncoder(spec, data.getColumnNames(), 1, null);
			encoder.build(data);
			FrameBlock meta = encoder.getMetaData(new FrameBlock(1, ValueType.STRING));
			
			//apply
			MultiColumnEncoder encoder2 = EncoderFactory.createEncoder(spec, data.getColumnNames(), 1, meta);
			MatrixBlock out = encoder2.apply(data2);
			
			//check outputs
			Assert.assertEquals(out.getNumRows(), data2.getNumRows());
			Assert.assertEquals(out.getNumColumns(), data2.getNumColumns());
			for(int i=1; i<=rows; i++)
				Assert.assertEquals(i, out.get(i-1, 0), 1e-8);
			for(int i=rows+1; i<=rows+10; i++)
				Assert.assertTrue(Double.isNaN(out.get(i-1, 0)));
		} 
		catch (DMLRuntimeException e) {
			e.printStackTrace();
			throw new RuntimeException(e);
		}
	}
	
	@Test
	public void testTransformApplyBinning() {
		try {
			//generate input data
			FrameBlock data = DataConverter.convertToFrameBlock(MatrixBlock.seqOperations(1, rows, 1));
			FrameBlock data2 = DataConverter.convertToFrameBlock(MatrixBlock.seqOperations(-5, rows+5, 1));
			
			//encode and obtain meta data
			String spec = "{ids:true, bin:[{id:1, method:equi-width, numbins:7}] }";
			MultiColumnEncoder encoder = EncoderFactory.createEncoder(spec, data.getColumnNames(), 1, null);
			encoder.build(data);
			FrameBlock meta = encoder.getMetaData(new FrameBlock(1, ValueType.STRING));
			
			//apply
			MultiColumnEncoder encoder2 = EncoderFactory.createEncoder(spec, data.getColumnNames(), 1, meta);
			MatrixBlock out = encoder2.apply(data2);
			
			//check outputs
			Assert.assertEquals(out.getNumRows(), data2.getNumRows());
			Assert.assertEquals(out.getNumColumns(), data2.getNumColumns());
			for(int i=-5; i<=rows+5; i++) {
				if( i < 1 )
					Assert.assertEquals(1, out.get(i+5, 0), 0.0);
				else if(i > rows)
					Assert.assertEquals(out.get(out.getNumRows()-1, 0), out.get(i+5, 0), 0.0);
				else
					Assert.assertEquals(((i-1)/10+1), out.get(i+5, 0), 1e-8);
			}
		}
		catch (Exception e) {
			e.printStackTrace();
			throw new RuntimeException(e);
		}
	}
}
