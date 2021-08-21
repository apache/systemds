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

package org.apache.sysds.test.functions.iogen;

import org.apache.sysds.runtime.io.MatrixReader;
import org.apache.sysds.runtime.iogen.GenerateReader;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestUtils;

public abstract class GenerateReaderTest extends AutomatedTestBase{

	protected final static String TEST_DIR = "functions/iogen/";
	protected final static String TEST_CLASS_DIR = TEST_DIR + GenerateReaderTest.class.getSimpleName() + "/";

	protected String sampleRaw;
	protected double[][] sampleMatrix;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
	}

	protected void generateRandomSymmetric(int size, double min, double max, double sparsity){
		sampleMatrix = getRandomMatrix(size, size, min, max, sparsity, 714);
		for(int i=0;i<size;i++) {
			for(int j = 0; j <= i; j++) {
				sampleMatrix[i][j] = sampleMatrix[j][i];
			}
		}
	}

	protected void runGenerateReaderTest() throws Exception {
		MatrixBlock sampleMatrixMB = DataConverter.convertToMatrixBlock(sampleMatrix);
		MatrixReader reader = GenerateReader.generateReader(sampleRaw, sampleMatrixMB);
	}
}
