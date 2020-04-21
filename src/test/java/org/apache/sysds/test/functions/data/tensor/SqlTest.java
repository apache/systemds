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

package org.apache.sysds.test.functions.data.tensor;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.Arrays;
import java.util.Collection;

@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class SqlTest extends AutomatedTestBase {
	private final static String TEST_DIR = "functions/data/";
	private final static String TEST_NAME = "SqlTest";
	private final static String TEST_CLASS_DIR = TEST_DIR + SqlTest.class.getSimpleName() + "/";
	private final static String DB_DROP_SUCCESS = "08006";
	private final static String DB_CONNECTION = "jdbc:derby:memory:derbyDB";
	private final static int DB_SIZE = 100;
	
	private String _query;
	
	public SqlTest(String query) {
		_query = query;
	}
	
	@Parameters
	public static Collection<Object[]> data() {
		Object[][] data = new Object[][]{
			{"SELECT * FROM test"},
			{"SELECT A.name, B.name FROM test A, test B"},
			{"SELECT COUNT(*) FROM test"},
			{"SELECT COUNT(*) FROM test A, test B"},
		};
		return Arrays.asList(data);
	}
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, null));
	}
	
	@Test
	public void sqlTestCP() {
		testSql(ExecMode.SINGLE_NODE);
	}
	
	@Test
	public void sqlTestHybrid() {
		testSql(ExecMode.HYBRID);
	}
	
	@Test
	public void sqlTestSpark() {
		testSql(ExecMode.SPARK);
	}
	
	private void testSql(ExecMode platform) {
		ExecMode platformOld = rtplatform;
		rtplatform = platform;
		
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == ExecMode.SPARK ) {
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		}
		Connection db = null;
		try {
			db = DriverManager.getConnection(DB_CONNECTION + ";create=true");
		}
		catch (SQLException e) {
			throw new RuntimeException(e);
		}
		try {
			//TODO test correctness
			//assertTrue("the test is not done, needs comparison, of result.", false);

			getAndLoadTestConfiguration(SqlTest.TEST_NAME);
			String HOME = SCRIPT_DIR + TEST_DIR;
			
			fullDMLScriptName = HOME + SqlTest.TEST_NAME + ".dml";
			
			programArgs = new String[]{"-explain", "-args", DB_CONNECTION, _query};
			
			Statement st = db.createStatement();
			st.execute("CREATE TABLE test(id INTEGER PRIMARY KEY, name VARCHAR(256), value DECIMAL(10,2))");
			StringBuilder sb = new StringBuilder("INSERT INTO test VALUES ");
			for (int i = 0; i < DB_SIZE; i++) {
				char letter = (char) ('a' + (i % ('z' - 'a' + 1)));
				sb.append("(").append(i).append(",'").append(letter)
					.append("',").append(i).append("),");
			}
			// remove last `,` char
			sb.setLength(sb.length() - 1);
			st.execute(sb.toString());
			// TODO check tensors (write not implemented yet, so not possible)
			runTest(true, false, null, -1);
		}
		catch (SQLException e) {
			throw new RuntimeException(e);
		}
		finally {
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
			try {
				DriverManager.getConnection(DB_CONNECTION + ";drop=true");
			}
			catch (SQLException e) {
				if( !e.getSQLState().equals(DB_DROP_SUCCESS) ) {
					throw new RuntimeException(e);
				}
			}
		}
	}
}
