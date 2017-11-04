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

package org.apache.sysml.test.integration.functions.jmlc;

import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.utils.lite.BuildLiteExecution;
import org.junit.Test;

public class BuildLiteJarTest extends AutomatedTestBase 
{
	@Override
	public void setUp() {
		//do nothing
	}
	
	@Test
	public void testJMLCHelloWorld() throws Exception {
		BuildLiteExecution.jmlcHelloWorld();
	}
	
	@Test
	public void testJMLCScoringExample() throws Exception {
		BuildLiteExecution.jmlcScoringExample();
	}
	
	@Test
	public void testJMLCUnivariateStats() throws Exception {
		BuildLiteExecution.jmlcUnivariateStatistics();
	}
	
	@Test
	public void testJMLCWriteReadMatrix() throws Exception {
		BuildLiteExecution.jmlcWriteMatrix();
		BuildLiteExecution.jmlcReadMatrix();
	}
	
	@Test
	public void testJMLCBasics() throws Exception {
		BuildLiteExecution.jmlcBasics();
	}
	
	@Test
	public void testJMLCL2SVM() throws Exception {
		BuildLiteExecution.jmlcL2SVM();
	}
	
	@Test
	public void testJMLCLinReg() throws Exception {
		BuildLiteExecution.jmlcLinReg();
	}
	
	@Test
	public void testJMLCALS() throws Exception {
		BuildLiteExecution.jmlcALS();
	}
	
	@Test
	public void testJMLCKmeans() throws Exception {
		BuildLiteExecution.jmlcKmeans();
	}
	
	@Test
	public void testJMLCTests() throws Exception {
		BuildLiteExecution.jmlcTests();
	}
}
