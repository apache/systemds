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

package org.apache.sysml.test.unit;


import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;

import org.apache.sysml.conf.DMLConfig;
import org.apache.sysml.parser.ParseException;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.instructions.gpu.context.GPUContextPool;
import org.junit.Assert;
import org.junit.Test;

/**
 * To test utility functions scattered throughout the codebase
 */
public class UtilsTest {

	@Test
	public void testParseListString0() {
		Assert.assertEquals(Arrays.asList(0), GPUContextPool.parseListString("0", 10));
	}

	@Test
	public void testParseListString1() {
		Assert.assertEquals(Arrays.asList(7), GPUContextPool.parseListString("7", 10));
	}

	@Test
	public void testParseListString2() {
		Assert.assertEquals(Arrays.asList(0,1,2,3), GPUContextPool.parseListString("-1", 4));
	}

	@Test
	public void testParseListString3() {
		Assert.assertEquals(Arrays.asList(0,1,2,3), GPUContextPool.parseListString("0,1,2,3", 6));
	}

	@Test
	public void testParseListString4() {
		Assert.assertEquals(Arrays.asList(0,1,2,3), GPUContextPool.parseListString("0-3", 6));
	}

	@Test(expected=IllegalArgumentException.class)
	public void testParseListStringFail0() {
		GPUContextPool.parseListString("7", 4);
	}

	@Test(expected=IllegalArgumentException.class)
	public void testParseListStringFail1() {
		GPUContextPool.parseListString("0,1,2,3", 2);
	}

	@Test(expected=IllegalArgumentException.class)
	public void testParseListStringFail2() {
		GPUContextPool.parseListString("0,1,2,3-4", 2);
	}

	@Test(expected=IllegalArgumentException.class)
	public void testParseListStringFail4() {
		GPUContextPool.parseListString("-1-4", 6);
	}


	@Test
	public void testDMLConfig1() throws DMLRuntimeException{
		DMLConfig dmlConfig = new DMLConfig();
		dmlConfig.setTextValue("A", "a");
		dmlConfig.setTextValue("B", "b");
		dmlConfig.setTextValue("C", "2");
		dmlConfig.setTextValue("D", "5");
		dmlConfig.setTextValue("E", "5.01");

		Assert.assertEquals("a", dmlConfig.getTextValue("A"));
		Assert.assertEquals("b", dmlConfig.getTextValue("B"));
		Assert.assertEquals(2, dmlConfig.getIntValue("C"));
		Assert.assertEquals(5, dmlConfig.getIntValue("D"));
		Assert.assertEquals(5.01, dmlConfig.getDoubleValue("E"), 1e-15);

		dmlConfig.setTextValue("E", "a");
		Assert.assertEquals("a", dmlConfig.getTextValue("E"));
	}



	@Test
	public void testDMLConfig2() throws DMLRuntimeException, IOException, ParseException {

		String testStr = "<root>"
				+ "<A>a</A>"
				+ "<B>b</B>"
				+ "<C>2</C>"
				+ "<D>5</D>"
				+ "<E>5.01</E>"
				+ "</root>";
		File temp = File.createTempFile("tempfile", null);
		BufferedWriter bw = new BufferedWriter(new FileWriter(temp));
		bw.write(testStr);
		bw.close();

		DMLConfig dmlConfig = new DMLConfig(temp.getAbsolutePath());

		Assert.assertEquals("a", dmlConfig.getTextValue("A"));
		Assert.assertEquals("b", dmlConfig.getTextValue("B"));
		Assert.assertEquals(2, dmlConfig.getIntValue("C"));
		Assert.assertEquals(5, dmlConfig.getIntValue("D"));
		Assert.assertEquals(5.01, dmlConfig.getDoubleValue("E"), 1e-15);

		dmlConfig.setTextValue("E", "a");
		Assert.assertEquals("a", dmlConfig.getTextValue("E"));
	}




}
