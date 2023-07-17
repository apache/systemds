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

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.codegen.CodegenUtils;
import org.apache.sysds.runtime.io.MatrixReader;
import org.apache.sysds.runtime.io.FrameReader;
import org.apache.sysds.runtime.iogen.codegen.FrameCodeGen;
import org.apache.sysds.runtime.iogen.codegen.MatrixCodeGen;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

import java.util.Random;

public abstract class GenerateReader {

	protected static final Log LOG = LogFactory.getLog(GenerateReader.class.getName());

	protected CustomProperties properties;

	protected FormatIdentifyer formatIdentifyer;

	public GenerateReader(SampleProperties sampleProperties) throws Exception {

		formatIdentifyer = sampleProperties.getDataType().isMatrix() ? new FormatIdentifyer(sampleProperties.getSampleRaw(),
			sampleProperties.getSampleMatrix()) : new FormatIdentifyer(sampleProperties.getSampleRaw(),
			sampleProperties.getSampleFrame());

		properties = formatIdentifyer.getFormatProperties();
		if(properties == null) {
			throw new Exception("The file format couldn't recognize!!");
		}
		if(sampleProperties.getDataType().isFrame()){
			properties.setSchema(sampleProperties.getSampleFrame().getSchema());
		}
	}

	public String getRandomClassName() {
		Random r = new Random();
		int low = 0;
		int high = 100000000;
		int result = r.nextInt(high - low) + low;

		return "GIOReader_" + result;
	}

	public CustomProperties getProperties() {
		return properties;
	}

	// Generate Reader for Matrix
	public static class GenerateReaderMatrix extends GenerateReader {

		private MatrixReader matrixReader;

		public GenerateReaderMatrix(SampleProperties sampleProperties) throws Exception {
			super(sampleProperties);
		}

		public GenerateReaderMatrix(String sampleRaw, MatrixBlock sampleMatrix, boolean parallel) throws Exception {
			super(new SampleProperties(sampleRaw, sampleMatrix));
			properties.setParallel(parallel);
		}

		public MatrixReader getReader() throws Exception {
			String className = getRandomClassName();
			MatrixCodeGen src = new MatrixCodeGen(properties, className);
			// constructor with arguments as CustomProperties
			Class[] cArg = new Class[1];
			cArg[0] = CustomProperties.class;
			String srcJava =  src.generateCodeJava(formatIdentifyer);
			matrixReader = (MatrixReader) CodegenUtils.compileClass(className, srcJava).getDeclaredConstructor(cArg).newInstance(properties);
			return matrixReader;
		}
	}

	// Generate Reader for Frame
	public static class GenerateReaderFrame extends GenerateReader {

		private FrameReader frameReader;

		public GenerateReaderFrame(SampleProperties sampleProperties) throws Exception {
			super(sampleProperties);
		}

		public GenerateReaderFrame(String sampleRaw, FrameBlock sampleFrame, boolean parallel) throws Exception {
			super(new SampleProperties(sampleRaw, sampleFrame));
			properties.setParallel(parallel);
		}

		public FrameReader getReader() throws Exception {
			String className = getRandomClassName();
			FrameCodeGen src = new FrameCodeGen(properties, className);
			// constructor with arguments as CustomProperties
			Class[] cArg = new Class[1];
			cArg[0] = CustomProperties.class;
			String srcJava = src.generateCodeJava(formatIdentifyer);
			frameReader = (FrameReader) CodegenUtils.compileClass(className, srcJava).getDeclaredConstructor(cArg).newInstance(properties);
			return frameReader;
		}
	}
}
