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

package org.apache.sysds.test.component.utils;

import java.io.IOException;
import java.util.HashMap;

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.hops.rewrite.HopDagValidator;
import org.apache.sysds.hops.rewrite.RewriteAlgebraicSimplificationStatic;
import org.apache.sysds.parser.DMLProgram;
import org.apache.sysds.parser.DMLTranslator;
import org.apache.sysds.parser.ParserFactory;
import org.apache.sysds.parser.ParserWrapper;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;

public class HopDagValidationTest extends AutomatedTestBase
{
	private static final String TEST_DIR = "component/parfor/";
	private static final String HOME = SCRIPT_DIR + TEST_DIR;
	private static final String TEST_CLASS_DIR = TEST_DIR + HopDagValidationTest.class.getSimpleName() + "/";
	
	@Override
	public void setUp() {}
	
	@Test
	public void testDependencyAnalysis1() { runTest("parfor1.dml"); }
	
	@Test
	public void testDependencyAnalysis3() { runTest("parfor3.dml"); }
	
	@Test
	public void testDependencyAnalysis4() { runTest("parfor4.dml"); }
	
	@Test
	public void testDependencyAnalysis6() { runTest("parfor6.dml"); }
	
	@Test
	public void testDependencyAnalysis7() { runTest("parfor7.dml"); }
	
	
	private void runTest( String scriptFilename ) {
		int index = scriptFilename.lastIndexOf(".dml");
		String testName = scriptFilename.substring(0, index > 0 ? index : scriptFilename.length());
		TestConfiguration testConfig = new TestConfiguration(TEST_CLASS_DIR, testName, new String[] {});
		addTestConfiguration(testName, testConfig);
		loadTestConfiguration(testConfig);
		
		try {
			DMLConfig conf = new DMLConfig(getCurConfigFile().getPath());
			ConfigurationManager.setLocalConfig(conf);
			
			//read script
			String dmlScriptString = DMLScript.readDMLScript(true, HOME + scriptFilename);
		
			//parsing and dependency analysis
			ParserWrapper parser = ParserFactory.createParser();
			DMLProgram prog = parser.parse(DMLScript.DML_FILE_PATH_ANTLR_PARSER, dmlScriptString, new HashMap<>());
			DMLTranslator dmlt = new DMLTranslator(prog);
			dmlt.validateParseTree(prog);
			HopDagValidator.validateHopDag(prog.getStatementBlocks().get(0).getHops(),
				new RewriteAlgebraicSimplificationStatic());
		}
		catch (IOException e) {
			e.printStackTrace();
			Assert.fail();
		}
	}
}
