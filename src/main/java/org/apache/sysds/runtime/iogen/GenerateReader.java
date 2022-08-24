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

import com.google.gson.Gson;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.codegen.CodegenUtils;
import org.apache.sysds.runtime.io.MatrixReader;
import org.apache.sysds.runtime.io.FrameReader;
import org.apache.sysds.runtime.iogen.codegen.FrameCodeGen;
import org.apache.sysds.runtime.iogen.codegen.MatrixCodeGen;

import java.io.File;
import java.io.FileWriter;

public abstract class GenerateReader {

	protected static final Log LOG = LogFactory.getLog(GenerateReader.class.getName());

	protected CustomProperties properties;
	protected String src;
	protected String className;

	public GenerateReader(SampleProperties sampleProperties) throws Exception {
		FormatIdentifyer formatIdentifyer = sampleProperties.getDataType().isMatrix() ? new FormatIdentifyer(sampleProperties.getSampleRaw(),
			sampleProperties.getSampleMatrix()) : new FormatIdentifyer(sampleProperties.getSampleRaw(), sampleProperties.getSampleFrame());

		properties = formatIdentifyer.getFormatProperties();
		if(properties == null) {
			throw new Exception("The file format couldn't recognize!!");
		}
		if(sampleProperties.getDataType().isFrame()) {
			properties.setSchema(sampleProperties.getSampleFrame().getSchema());
		}
		properties.setParallel(sampleProperties.isParallel());

		String[] path = sampleProperties.getFormat().split("/");
		String fileName = path[path.length - 1];
		if(path.length > 1) {
			String dirPath = sampleProperties.getFormat().substring(0, sampleProperties.getFormat().length() - fileName.length());
			File outDir = new File(dirPath);
			outDir.getParentFile().mkdirs();
		}
		className = fileName.split("\\.")[0];
		String srcJava = getReaderString();
		FileWriter srcWriter = new FileWriter(sampleProperties.getFormat());
		srcWriter.write(srcJava);
		srcWriter.close();

		Gson gson = new Gson();
		FileWriter propWriter = new FileWriter(sampleProperties.getFormat() + ".prop");
		propWriter.write(gson.toJson(properties));
		propWriter.close();
	}

	public GenerateReader(CustomProperties properties, String src, String className) {
		this.properties = properties;
		this.src = src;
		this.className = className;
	}

	public CustomProperties getProperties() {
		return properties;
	}

	public abstract String getReaderString();

	// Generate Reader for Matrix
	public static class GenerateReaderMatrix extends GenerateReader {

		private MatrixReader matrixReader;

		public GenerateReaderMatrix(SampleProperties sampleProperties) throws Exception {
			super(sampleProperties);
		}

		public GenerateReaderMatrix(CustomProperties properties, String src, String className) {
			super(properties, src, className);
		}

		public MatrixReader getReader() throws Exception {
			Class[] cArg = new Class[1];
			cArg[0] = CustomProperties.class;
			matrixReader = (MatrixReader) CodegenUtils.compileClass(className, src).getDeclaredConstructor(cArg).newInstance(properties);
			return matrixReader;
		}

		@Override
		public String getReaderString() {
			MatrixCodeGen src = new MatrixCodeGen(properties, className);
			// constructor with arguments as CustomProperties
			Class[] cArg = new Class[1];
			cArg[0] = CustomProperties.class;
			String srcJava = src.generateCodeJava();
			return srcJava;
		}
	}

	// Generate Reader for Frame
	public static class GenerateReaderFrame extends GenerateReader {

		private FrameReader frameReader;

		public GenerateReaderFrame(SampleProperties sampleProperties) throws Exception {
			super(sampleProperties);
		}

		public GenerateReaderFrame(CustomProperties properties, String src, String className) {
			super(properties, src, className);
		}

		public FrameReader getReader() throws Exception {
			Class[] cArg = new Class[1];
			cArg[0] = CustomProperties.class;
			frameReader = (FrameReader) CodegenUtils.compileClass(className, src).getDeclaredConstructor(cArg).newInstance(properties);
			return frameReader;
		}

		@Override
		public String getReaderString() {
			FrameCodeGen src = new FrameCodeGen(properties, className);
			// constructor with arguments as CustomProperties
			Class[] cArg = new Class[1];
			cArg[0] = CustomProperties.class;
			String srcJava = src.generateCodeJava();
			return srcJava;
		}
	}
}
