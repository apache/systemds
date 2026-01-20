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
package org.apache.sysds.test.component.matrix.nativeBLAS;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import java.io.File;
import java.lang.Math;
import java.lang.String;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.utils.NativeHelper;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.test.component.matrix.MatrixMultiplyTest;

public class MatrixMultiplyMKLTest extends MatrixMultiplyTest{
	protected static final Log LOG = LogFactory.getLog(MatrixMultiplyTest.class.getName());

	private final static String TEST_DIR = "component/matrix/nativeBLAS/";

	protected String getTestClassDir() {
		return getTestDir() + this.getClass().getSimpleName() + "/";
	}

	protected String getTestName() {
		return "matrixMultiplyMKL";
	}

	protected String getTestDir() {
		return TEST_DIR;
	}

    public MatrixMultiplyMKLTest(int i, int j, int k, double s, double s2, int p, boolean self) {
		super(i,j,k,s,s2,p,self);
	}
	
	// TODO: test not working with native BLAS
	@Override
	public void testLeftNonContiguous(){}

	@Override
	protected File getConfigTemplateFile() {
		return new File("./src/test/config/component/matrix/nativeBLAS/SystemDS-config-MKL.xml");
	}

	@Override
	public void setUp() {
		try {
			String testname = getTestName() + String.valueOf(Math.random()*5);
			addTestConfiguration(testname, new TestConfiguration(getTestClassDir(), testname));
			loadTestConfiguration(getTestConfiguration(testname));

			DMLConfig conf = new DMLConfig(getCurConfigFile().getPath());
			ConfigurationManager.setLocalConfig(conf);

			assertEquals(true, NativeHelper.isNativeLibraryLoaded());
		} catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Override
	public void tearDown() {
		TestUtils.clearDirectory(getCurLocalTempDir().getPath());
		TestUtils.removeDirectories(new String[]{getCurLocalTempDir().getPath()});
	}
}
