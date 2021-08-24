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

import org.apache.sysds.runtime.io.*;
import org.apache.sysds.runtime.iogen.GenerateReader;
import org.apache.sysds.runtime.iogen.MatrixGenerateReader;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestUtils;

import java.io.BufferedWriter;
import java.io.FileWriter;

public abstract class GenerateReaderTest extends AutomatedTestBase {

	protected final static String TEST_DIR = "functions/iogen/";
	protected final static String TEST_CLASS_DIR = TEST_DIR + GenerateReaderTest.class.getSimpleName() + "/";

	protected String sampleRaw;
	protected double[][] sampleMatrix;

	@Override public void setUp() {
		TestUtils.clearAssertionInformation();
	}

	protected void generateRandomSymmetric(int size, double min, double max, double sparsity, boolean isSkew) {
		sampleMatrix = getRandomMatrix(size, size, min, max, sparsity, 714);
		int conf = isSkew ? -1 : 1;
		for(int i = 0; i < size; i++) {
			for(int j = 0; j <= i; j++) {

				if(i != j)
					sampleMatrix[i][j] = sampleMatrix[j][i] * conf;
				else
					sampleMatrix[i][j] = 0;
			}
		}
	}

	protected void runGenerateReaderTest() throws Exception {
		MatrixBlock sampleMatrixMB = DataConverter.convertToMatrixBlock(sampleMatrix);
		GenerateReader gr = new GenerateReader(sampleRaw, sampleMatrixMB);
		MatrixReader reader=gr.getMatrixReader();

		// Write SampleRawMatrix data into a file
		String HOME = SCRIPT_DIR + TEST_DIR;
		String outName = HOME + OUTPUT_DIR ;
		String fileNameSampleRaw = outName+"/SampleRaw.txt";
		String fileNameSampleRawOut = outName+"/SampleRawOut.txt";
		BufferedWriter writer = new BufferedWriter(new FileWriter(fileNameSampleRaw));
		writer.write(sampleRaw);
		writer.close();

		// Read the data with Generated Reader
		MatrixBlock src = reader
			.readMatrixFromHDFS(fileNameSampleRaw, sampleMatrixMB.getNumRows(), sampleMatrixMB.getNumColumns(), -1, -1);

		if(this instanceof MatrixGenerateReaderCSVTest) {
			FileFormatPropertiesCSV csv = new FileFormatPropertiesCSV(false, ",", false);
			WriterTextCSV writerTextCSV = new WriterTextCSV(csv);
			writerTextCSV.writeMatrixToHDFS(src, fileNameSampleRawOut, src.getNumRows(), src.getNumColumns(), -1,
				src.getNonZeros(), false);
		}
		else if(this instanceof MatrixGenerateReaderLibSVMTest){
			FileFormatPropertiesLIBSVM libsvm = new FileFormatPropertiesLIBSVM(" ", ":", false);
			WriterTextLIBSVM writerTextLIBSVM = new WriterTextLIBSVM(libsvm);
			writerTextLIBSVM.writeMatrixToHDFS(src, fileNameSampleRawOut, src.getNumRows(), src.getNumColumns(), -1,
				src.getNonZeros(), false);
		}
		else if(this instanceof MatrixMatrixGenerateReaderMarketTest){
			WriterTextCell writerTextCell = new WriterTextCell();
			writerTextCell.writeMatrixToHDFS(src, fileNameSampleRawOut, src.getNumRows(), src.getNumColumns(), -1,
				src.getNonZeros(), false);
		}


	}
}
