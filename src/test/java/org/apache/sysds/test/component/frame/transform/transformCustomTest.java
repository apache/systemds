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

package org.apache.sysds.test.component.frame.transform;

import static org.junit.Assert.fail;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.transform.encode.EncoderFactory;
import org.apache.sysds.runtime.transform.encode.MultiColumnEncoder;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public class transformCustomTest {
	protected static final Log LOG = LogFactory.getLog(transformCustomTest.class.getName());

	final FrameBlock data;

	public transformCustomTest() {
		data = TestUtils.generateRandomFrameBlock(100, new ValueType[] {ValueType.UINT8}, 231);
		data.setSchema(new ValueType[] {ValueType.INT32});
	}

	@Test
	public void testRecode() {
		test("{recode:[C1]}");
	}

	@Test
	public void testBin() {
		test("{ids:true, bin:[{id:1, method:equi-width, numbins:4}]}");
	}

	@Test
	public void testBin2() {
		test("{ids:true, bin:[{id:1, method:equi-width, numbins:100}]}");
	}

	@Test
	public void testBin3() {
		test("{ids:true, bin:[{id:1, method:equi-width, numbins:2}]}");
	}

	@Test
	public void testBin4() {
		test("{ids:true, bin:[{id:1, method:equi-height, numbins:2}]}");
	}

	@Test
	public void testBin5() {
		test("{ids:true, bin:[{id:1, method:equi-height, numbins:10}]}");
	}

	public void test(String spec) {
		try {

			FrameBlock meta = null;
			MultiColumnEncoder encoder = EncoderFactory.createEncoder(spec, data.getColumnNames(), data.getNumColumns(),
				meta);
			MatrixBlock out = encoder.encode(data);
			MatrixBlock out2 = encoder.apply(data);

			TestUtils.compareMatrices(out, out2, 0, "Not Equal after apply");
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}
}
