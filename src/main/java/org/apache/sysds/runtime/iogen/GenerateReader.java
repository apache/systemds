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

package org.apache.sysds.runtime.iogen;

import org.apache.sysds.runtime.io.FrameReader;
import org.apache.sysds.runtime.io.MatrixReader;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.nio.file.Files;
import java.nio.file.Paths;

/*
	Generate Reader has two steps:
		1. Identify file format and extract the properties of it based on the Sample Matrix.
		 The ReaderMapping class tries to map the Sample Matrix on the Sample Raw Matrix.
		 The result of a ReaderMapping is a FileFormatProperties object.

		2. Generate a reader based on inferred properties.

		Note. Base on this implementation, it is possible to generate a reader base on Sample Matrix
		and generate a reader for a frame or vice versa.
	 */

public class GenerateReader {

	private ReaderMapping readerMapping;

	private MatrixReader matrixReader;

	private FrameReader frameReader;

	private SampleProperties sampleProperties;

	public GenerateReader(SampleProperties sampleProperties) throws Exception {

		this.sampleProperties = sampleProperties;

		//Read sample raw file into sampleRaw
		String sampleRaw = new String(Files.readAllBytes(Paths.get(sampleProperties.getSampleRawFileName())));
		if(sampleProperties.getDataType().isMatrix()) {
			MatrixBlock sampleMatrix = binaryDeserialization( Files.readAllBytes(Paths.get(sampleProperties.getSampleBinaryFileName())));
			readerMapping = generateReader(sampleRaw, sampleMatrix);
		}
		else if(sampleProperties.getDataType().isFrame()) {

		}

	}

	public GenerateReader(String sampleRaw, MatrixBlock sampleMatrix) throws Exception {
		readerMapping = this.generateReader(sampleRaw, sampleMatrix);
	}

	public GenerateReader(String sampleRaw, FrameBlock sampleFrame) {
		readerMapping = this.generateReader(sampleRaw, sampleFrame);
	}

	private ReaderMapping generateReader(String sampleRaw, MatrixBlock sampleMatrix) throws Exception {
		// Identify file format properties:
		return new ReaderMapping(sampleRaw, sampleMatrix);
	}

	private ReaderMapping generateReader(String sampleRaw, FrameBlock sampleFrame) {
		// TODO: extend mapping to support frame
		return null;
	}

	public MatrixReader getMatrixReader() throws Exception {

		boolean isMapped = readerMapping != null && readerMapping.isMapped();
		if(!isMapped) {
			throw new Exception("Sample raw data and sample matrix don't match !!");
		}
		CustomProperties ffp = readerMapping.getFormatProperties();
		if(ffp == null) {
			throw new Exception("The file format couldn't recognize!!");
		}
		ffp.setClen(sampleProperties.getSampleCols());
		// 2. Generate a Matrix Reader:
		if(ffp.getRowPattern().equals(CustomProperties.GRPattern.Regular)) {
			if(ffp.getColPattern().equals(CustomProperties.GRPattern.Regular)) {
				matrixReader = new MatrixGenerateReader.MatrixReaderRowRegularColRegular(ffp);
			}
			else {
				matrixReader = new MatrixGenerateReader.MatrixReaderRowRegularColIrregular(ffp);
			}
		}
		else {
			matrixReader = new MatrixGenerateReader.MatrixReaderRowIrregular(ffp);
		}
		return matrixReader;
	}

	public FrameReader getFrameReader() {
		return frameReader;
	}

	private MatrixBlock binaryDeserialization(byte[] buf) throws Exception {
		ByteArrayInputStream bis = new ByteArrayInputStream(buf);
		ObjectInputStream objectInputStream;
		MatrixBlock result = new MatrixBlock(sampleProperties.getSampleRows(), sampleProperties.getSampleCols(), false);
		try {
			objectInputStream = new ObjectInputStream(bis);
			for(int r = 0; r < sampleProperties.getSampleRows(); r++)
				for(int c = 0; c < sampleProperties.getSampleCols(); c++) {
					result.setValue(r, c, objectInputStream.readDouble());
				}
		}
		catch(IOException e) {
			throw new Exception("Can't read matrix from byteArray", e);
		}
		return result;
	}

}
