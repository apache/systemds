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
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
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
	public static final String META = "{\"data_type\": \"matrix\",\n" +
			"    \"value_type\": \"double\",  \n" +
			"    \"rows\": 1,\n" +
			"    \"cols\": 1,\n" +
			"    \"nnz\": 1,\n" +
			"    \"format\": \"csv\"}";
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
		boolean oldStat = DMLScript.STATISTICS;
		boolean oldJMLCStat = DMLScript.JMLC_MEM_STATISTICS;
        try (Connection conn = new Connection()) {
			DMLScript.STATISTICS = true;
			conn.gatherMemStats(true);
			Assert.assertTrue(DMLScript.STATISTICS);

            conn.prepareScript("printx('hello')", new String[]{}, new String[]{});
            throw new AssertionError("Test should have thrown a DMLException");
        } catch (DMLException e) {
            Throwable cause = e.getCause();
            Assert.assertTrue(cause.getMessage().startsWith("ERROR: [line 1:0] -> printx('hello') -- function printx is undefined in namespace .builtinNS"));
        } finally {
			DMLScript.STATISTICS = oldStat;
			DMLScript.JMLC_MEM_STATISTICS = oldJMLCStat;
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
	public void testReadDoubleMatrixException() {
		try (Connection conn = new Connection()) {
			conn.readDoubleMatrix("test.csv");
		} catch (IOException e) {
			Assert.assertEquals("IOException", e.getClass().getSimpleName());
		}
	}

	@Test
	public void testReadDoubleMatrixException2() {
		try (Connection conn = new Connection()) {
			conn.readDoubleMatrix("test.csv", null, 1, 1, 1, 1);
		} catch (IOException e) {
			Assert.assertEquals("IOException", e.getClass().getSimpleName());
		}
	}

	@Test
	public void testReadDoubleMatrix() {
		try (Connection conn = new Connection()) {
			double[][] matrix = conn.readDoubleMatrix("src/test/resources/component/compress/1-1.csv");
			Assert.assertEquals(1.0, matrix[0][0], 0.0);
		} catch (IOException e) {
			throw new AssertionError(e);
		}
	}

	@Test
	public void testReadDoubleMatrix2() {
		try (Connection conn = new Connection()) {
			double[][] matrix = conn.readDoubleMatrix("src/test/resources/component/compress/1-1.csv", Types.FileFormat.CSV, 1, 1, 1, 1);
			Assert.assertEquals(1.0, matrix[0][0], 0.0);
		} catch (IOException e) {
			throw new AssertionError(e);
		}
	}

	@Test
	public void testConvertToDoubleMatrix() {
		try (Connection conn = new Connection()) {
			double[][] matrix = conn.convertToDoubleMatrix("1", META);
			Assert.assertEquals(1.0, matrix[0][0], 0.0);
		} catch (IOException e) {
			throw new AssertionError(e);
		}
	}

	@Test
	public void testConvertToDoubleMatrixException() {
		try (Connection conn = new Connection()) {
			String meta = "abc";
			conn.convertToDoubleMatrix("1", meta);
		} catch (IOException e) {
			Throwable cause = e.getCause();
			Assert.assertEquals("NullPointerException", cause.getClass().getSimpleName());
		}
	}

	@Test
	public void testConvertToMatrix1() {
		try (Connection conn = new Connection()) {
			MatrixBlock mb = conn.convertToMatrix("1", META);
			Assert.assertEquals(1.0, mb.get(0,0), 0.0);
		} catch (IOException e) {
			throw new AssertionError(e);
		}
	}

	@Test
	public void testConvertToMatrixException1() {
		try (Connection conn = new Connection()) {
			conn.convertToMatrix("1", "{" + META);
		} catch (IOException e) {
			Throwable cause = e.getCause();
			Assert.assertEquals("NullPointerException", cause.getClass().getSimpleName());
		}
	}

	@Test
	public void testConvertToMatrix2() {
		try (Connection conn = new Connection()) {
			MatrixBlock mb = conn.convertToMatrix("1 1 1", 1, 1);
			Assert.assertEquals(1.0, mb.get(0,0), 0.0);
		} catch (IOException e) {
			throw new AssertionError(e);
		}
	}

	@Test
	public void testConvertToMatrixException2() {
		try (Connection conn = new Connection()) {
			conn.convertToMatrix("1 1", 1,1);
		} catch (IOException e) {
			Throwable cause = e.getCause();
			Assert.assertEquals("StringIndexOutOfBoundsException", cause.getClass().getSimpleName());
		}
	}

	@Test
	public void testConvertToMatrixCSV() {
		try (Connection conn = new Connection()) {
			MatrixBlock mb = conn.convertToMatrix(IOUtilFunctions.toInputStream("1"), 1,1,"csv");
			Assert.assertEquals(1.0, mb.get(0,0), 0.0);
		} catch (IOException e) {
			throw new AssertionError(e);
		}
	}

	@Test
	public void testConvertToMatrixMM() {
		try (Connection conn = new Connection()) {
			MatrixBlock mb = conn.convertToMatrix(IOUtilFunctions.toInputStream("%%MatrixMarket matrix coordinate real"+
					" general \n 1 1 1 \n 1 1 1.0"), 1,1,"mm");
			Assert.assertEquals(1.0, mb.get(0,0), 0.0);
		} catch (IOException e) {
			throw new AssertionError(e);
		}
	}

	@Test
	public void testConvertToMatrixInvalidFormat() {
		try (Connection conn = new Connection()) {
			conn.convertToMatrix(IOUtilFunctions.toInputStream("1"), 1,1,"abc");
		} catch (IOException e) {
			Assert.assertTrue(e.getMessage().startsWith("Invalid input format"));
		}
	}

	@Test
	public void testReadFrame1() {
		try (Connection conn = new Connection()) {
			conn.readStringFrame("test.csv");
		} catch (IOException e) {
			Assert.assertEquals("IOException", e.getCause().getClass().getSimpleName());
		}
	}

	@Test
	public void testReadFrame2() {
		try (Connection conn = new Connection()) {
			conn.readStringFrame("test.csv", Types.FileFormat.CSV, 1, 1);
		} catch (IOException e) {
			Assert.assertTrue(e.getCause().getMessage().startsWith("File test.csv does not exist on HDFS/LFS"));
		}
	}

	@Test
	public void testConvertToStringFrame1() {
		try (Connection conn = new Connection()) {
			String[][] frame = conn.convertToStringFrame("Hello", META);
			Assert.assertEquals("Hello", frame[0][0]);
		} catch (IOException e) {
			throw new AssertionError(e);
		}
	}

	@Test
	public void testConvertToStringFrameException1() {
		try (Connection conn = new Connection()) {
			conn.convertToFrame("Hello", "{" + META);
		} catch (IOException e) {
			Assert.assertEquals("NullPointerException", e.getCause().getClass().getSimpleName());
		}
	}

	@Test
	public void testConvertToStringFrame2() {
		try (Connection conn = new Connection()) {
			String[][] frame = conn.convertToStringFrame("1 1 Hello", 1,1);
			Assert.assertEquals("Hello", frame[0][0]);
		} catch (IOException e) {
			throw new AssertionError(e);
		}
	}

	@Test
	public void testConvertToStringFrame3() {
		try (Connection conn = new Connection()) {
			String[][] frame = conn.convertToStringFrame(IOUtilFunctions.toInputStream("Hello"), 1,1, "csv");
			Assert.assertEquals("Hello", frame[0][0]);
		} catch (IOException e) {
			throw new AssertionError(e);
		}
	}

	@Test
	public void testConvertToStringFrame4() {
		try (Connection conn = new Connection()) {
			String[][] frame = conn.convertToStringFrame(IOUtilFunctions.toInputStream("%%MatrixMarket matrix coordinate real"+
					" general \n 1 1 1 \n 1 1 Hello"), 1,1, "mm");
			Assert.assertEquals("Hello", frame[0][0]);
		} catch (IOException e) {
			Assert.assertEquals("Failed to create frame reader for unknown format: mm", e.getCause().getMessage());
		}
	}

	@Test
	public void testConvertToStringFrameException2() {
		try (Connection conn = new Connection()) {
			String[][] frame = conn.convertToStringFrame(IOUtilFunctions.toInputStream("Hi"), 1,1, "abc");
			Assert.assertEquals("Hello", frame[0][0]);
		} catch (IOException e) {
			Assert.assertTrue(e.getMessage().startsWith("Invalid input format"));
		}
	}
}
