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
 
package org.apache.sysds.test.functions.jmlc;

import java.io.IOException;

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.api.DMLException;
import org.apache.sysds.api.jmlc.Connection;
import org.apache.sysds.api.jmlc.PreparedScript;
import org.apache.sysds.api.jmlc.ResultVariables;
import org.apache.sysds.conf.CompilerConfig.ConfigType;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.AutomatedTestBase;

public class ResultVariableTest extends AutomatedTestBase 
{
	@Override
	public void setUp() {}
	
	@Test
	public void testMatrix() throws IOException {
		Connection conn = new Connection();
		PreparedScript ps = conn.prepareScript(
			"R=as.matrix(7);\nwrite(R, \"xyy\");", new String[] {}, new String[] {"R"});
		ResultVariables ret = ps.executeScript();
		Assert.assertTrue(ret.getMatrix("R") instanceof double[][]);
		Assert.assertTrue(ret.getMatrixBlock("R") instanceof MatrixBlock);
		try{ ret.getMatrix("xR"); } catch(DMLException ex)
			{Assert.assertTrue(ex.getMessage().startsWith("Non-existent"));}
		try{ ret.getMatrixBlock("xR"); } catch(DMLException ex)
			{Assert.assertTrue(ex.getMessage().startsWith("Non-existent"));}
		try{ ret.getFrame("R"); } catch(DMLException ex)
			{Assert.assertTrue(ex.getMessage().startsWith("Expected frame result"));}
		try{ ret.getFrameBlock("R"); } catch(DMLException ex)
			{Assert.assertTrue(ex.getMessage().startsWith("Expected frame result"));}
		try{ ret.getList("R"); } catch(DMLException ex)
			{Assert.assertTrue(ex.getMessage().startsWith("Expected list result"));}
		try{ ret.getListObject("R"); } catch(DMLException ex)
			{Assert.assertTrue(ex.getMessage().startsWith("Expected list result"));}
		conn.close();
	}
	
	@Test
	public void testFrame() throws IOException {
		Connection conn = new Connection();
		PreparedScript ps = conn.prepareScript(
			"R=as.frame(7);\nwrite(R, \"xyy\");", new String[] {}, new String[] {"R"});
		ResultVariables ret = ps.executeScript();
		Assert.assertTrue(ret.getFrame("R") instanceof String[][]);
		Assert.assertTrue(ret.getFrameBlock("R") instanceof FrameBlock);
		try{ ret.getFrame("xR"); } catch(DMLException ex)
			{Assert.assertTrue(ex.getMessage().startsWith("Non-existent"));}
		try{ ret.getFrameBlock("xR"); } catch(DMLException ex)
			{Assert.assertTrue(ex.getMessage().startsWith("Non-existent"));}
		try{ ret.getMatrix("R"); } catch(DMLException ex)
			{Assert.assertTrue(ex.getMessage().startsWith("Expected matrix result"));}
		try{ ret.getMatrixBlock("R"); } catch(DMLException ex)
			{Assert.assertTrue(ex.getMessage().startsWith("Expected matrix result"));}
		try{ ret.getList("R"); } catch(DMLException ex)
			{Assert.assertTrue(ex.getMessage().startsWith("Expected list result"));}
		try{ ret.getListObject("R"); } catch(DMLException ex)
			{Assert.assertTrue(ex.getMessage().startsWith("Expected list result"));}
		conn.close();
	}
	
	@Test
	public void testListAndMeta() throws IOException {
		Connection conn = new Connection(new DMLConfig(), ConfigType.CODEGEN_ENABLED);
		conn.setStatistics(true);
		conn.setLineage(true);
		PreparedScript ps = conn.prepareScript(
			"R=list(3,7);\nwrite(R, \"xyy\");", new String[] {}, new String[] {"R"});
		ResultVariables ret = ps.executeScript();
		Assert.assertTrue(ret.size()==1);
		Assert.assertTrue(ret.getVariableNames().contains("R"));
		Assert.assertTrue(ret.getList("R").get(0) instanceof ScalarObject);
		Assert.assertTrue(ret.getListObject("R").getData(0) instanceof ScalarObject);
		try{ ret.getScalarObject("xR"); } catch(DMLException ex)
			{Assert.assertTrue(ex.getMessage().startsWith("Non-existent"));}
		try{ ret.getScalarObject("R"); } catch(DMLException ex)
			{Assert.assertTrue(ex.getMessage().startsWith("Expected scalar result"));}
		conn.close();
	}
}
