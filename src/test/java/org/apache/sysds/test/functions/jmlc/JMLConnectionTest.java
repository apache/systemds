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

import org.apache.sysds.api.DMLException;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.api.jmlc.Connection;
import org.apache.sysds.api.jmlc.PreparedScript;
import org.apache.sysds.common.Types;
import org.apache.sysds.conf.CompilerConfig;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.parser.LanguageException;
import org.apache.sysds.test.AutomatedTestBase;
import org.junit.Assert;
import org.junit.Test;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.HashMap;

/**
 * Test input and output capabilities of JMLC Connection class.
 */
public class JMLConnectionTest extends AutomatedTestBase {
	private final static String TEST_NAME = "JMLConnectionTest";
	private final static String TEST_DIR = "functions/jmlc/";

	@Override
	public void setUp() {
		addTestConfiguration(TEST_DIR, TEST_NAME);
		getAndLoadTestConfiguration(TEST_NAME);
	}

	@Test
	public void testConnectionInvalidArgName() throws DMLException {
		Connection conn = new Connection();
		boolean oldStat = DMLScript.STATISTICS;
		boolean oldJMLCStat = DMLScript.JMLC_MEM_STATISTICS;
		conn.gatherMemStats(true);
		Assert.assertTrue(DMLScript.STATISTICS);
		Assert.assertTrue(DMLScript.JMLC_MEM_STATISTICS);

		HashMap<String,String> args = new HashMap<>();
		args.put("TEST", "TEST");
		args.put(null, "TEST");
		try {
			conn.prepareScript("print('hello')", args, new String[] {}, new String[] {});
			throw new AssertionError("Test should have thrown a LanguageException");
		} catch (LanguageException e) {
			Assert.assertTrue(e.getMessage().startsWith("Invalid argument names"));
		} finally {
			conn.close();
			DMLScript.STATISTICS = oldStat;
			DMLScript.JMLC_MEM_STATISTICS = oldJMLCStat;
		}
	}

	@Test
	public void testConnectionInvalidInName() throws DMLException {
		Connection conn = new Connection();

		boolean oldStat = DMLScript.STATISTICS;
		boolean oldJMLCStat = DMLScript.JMLC_MEM_STATISTICS;

		DMLScript.STATISTICS = true;
		conn.gatherMemStats(false);
		Assert.assertTrue(DMLScript.STATISTICS);

        try (conn) {
            conn.prepareScript("printx('hello')", new String[]{"$inScalar1", null}, new String[]{null});
            throw new AssertionError("Test should have thrown a LanguageException");
        } catch (LanguageException e) {
            Assert.assertTrue(e.getMessage().startsWith("Invalid variable names"));
        } finally {
            DMLScript.STATISTICS = oldStat;
            DMLScript.JMLC_MEM_STATISTICS = oldJMLCStat;
        }
	}

	@Test
	public void testConnectionParseLanguageException() {
        try (Connection conn = new Connection()) {
            conn.prepareScript("printx('hello')", new String[]{}, new String[]{});
            throw new AssertionError("Test should have thrown a DMLException");
        } catch (DMLException e) {
            Throwable cause = e.getCause();
            Assert.assertTrue(cause.getMessage().startsWith("ERROR: [line 1:0] -> printx('hello') -- function printx is undefined in namespace .builtinNS"));
        }
	}

	@Test
	public void testConnectionParseException() {
        try (Connection conn = new Connection()) {
            conn.prepareScript("print('hello'", new String[]{}, new String[]{});
            throw new AssertionError("Test should have thrown a ParseException");
        } catch (Exception e) {
            Assert.assertEquals("ParseException", e.getClass().getSimpleName());
        }
	}

	@Test
	public void testConnectionClose() {
		Connection conn = new Connection();
		CompilerConfig old = ConfigurationManager.getCompilerConfig();
		CompilerConfig tmp = new CompilerConfig();
		tmp.set(CompilerConfig.ConfigType.CODEGEN_ENABLED, true);
		ConfigurationManager.setGlobalConfig(tmp);
		conn.close();
		ConfigurationManager.setGlobalConfig(old);
	}

	@Test
	public void testReadScriptHDFS() {
        try (Connection conn = new Connection()) {
            conn.readScript("hdfs://localhost:9000/Test");
        } catch (IOException e) {
			Assert.assertEquals("ConnectException",e.getClass().getSimpleName());
        }
	}

	@Test
	public void testReadScriptGPFS() {
		try (Connection conn = new Connection()) {
			conn.readScript("gpfs://localhost:9000/Test");
		} catch (IOException e) {
			Assert.assertEquals("UnsupportedFileSystemException",e.getClass().getSimpleName());
		}
	}

	@Test
	public void testReadScriptS3() {
		try (Connection conn = new Connection()) {
			conn.readScript("s3://bucket/Test");
		} catch (IOException e) {
			Assert.assertEquals("UnsupportedFileSystemException",e.getClass().getSimpleName());
		}
	}

	@Test
	public void testDoubleMatrixException() {
		try (Connection conn = new Connection()) {
			conn.readDoubleMatrix("test.csv");
		} catch (IOException e) {
			Assert.assertEquals("IOException", e.getClass().getSimpleName());
		}
	}

	@Test
	public void testDoubleMatrixException2() {
		try (Connection conn = new Connection()) {
			conn.readDoubleMatrix("test.csv", null, 1, 1, 1, 1);
		} catch (IOException e) {
			Assert.assertEquals("IOException", e.getClass().getSimpleName());
		}
	}

	@Test
	public void testDoubleMatrix() {
		try (Connection conn = new Connection()) {
			double[][] matrix = conn.readDoubleMatrix("src/test/resources/component/compress/1-1.csv");
			Assert.assertEquals(1.0, matrix[0][0], 0.0);
		} catch (IOException e) {
			throw new AssertionError(e);
		}
	}

	@Test
	public void testDoubleMatrix2() {
		try (Connection conn = new Connection()) {
			double[][] matrix = conn.readDoubleMatrix("src/test/resources/component/compress/1-1.csv", Types.FileFormat.CSV, 1, 1, 1, 1);
			Assert.assertEquals(1.0, matrix[0][0], 0.0);
		} catch (IOException e) {
			throw new AssertionError(e);
		}
	}

	@Test
	public void testConvertMatrix() {
		try (Connection conn = new Connection()) {
			double[][] matrix = conn.convertToDoubleMatrix("src/test/resources/component/compress/1-1.csv", "tmp");
			Assert.assertEquals(1.0, matrix[0][0], 0.0);
		} catch (IOException e) {
			throw new AssertionError(e);
		}
	}
}
