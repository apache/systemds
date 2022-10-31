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
import org.apache.sysds.runtime.io.MatrixReader;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.io.FrameReader;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

/*
   Generate Reader has two steps:
      1. Identify file format and extract the properties of it based on the Sample Matrix.
      The ReaderMapping class tries to map the Sample Matrix on the Sample Raw Matrix.
      The result of a ReaderMapping is a FileFormatProperties object.

      2. Generate a reader based on inferred properties.

    Note. Base on this implementation, it is possible to generate a reader 
    base on Sample Matrix and generate a reader for a frame or vice versa.
*/
public abstract class GenerateReader {

	protected static final Log LOG = LogFactory.getLog(GenerateReader.class.getName());

	protected static ReaderMapping readerMapping;

	public GenerateReader(SampleProperties sampleProperties) throws Exception {

		readerMapping = sampleProperties.getDataType().isMatrix() ? new ReaderMapping.MatrixReaderMapping(
			sampleProperties.getSampleRaw(), sampleProperties.getSampleMatrix()) : new ReaderMapping.FrameReaderMapping(
			sampleProperties.getSampleRaw(), sampleProperties.getSampleFrame());
	}

	// Generate Reader for Matrix
	public static class GenerateReaderMatrix extends GenerateReader {

		private MatrixReader matrixReader;

		public GenerateReaderMatrix(SampleProperties sampleProperties) throws Exception {
			super(sampleProperties);
		}

		public GenerateReaderMatrix(String sampleRaw, MatrixBlock sampleMatrix) throws Exception {
			super(new SampleProperties(sampleRaw, sampleMatrix));
		}

		public MatrixReader getReader() throws Exception {

			boolean isMapped = readerMapping != null && readerMapping.isMapped();
			if(!isMapped) {
				throw new Exception("Sample raw data and sample matrix don't match !!");
			}
			CustomProperties ffp = readerMapping.getFormatProperties();
			if(ffp == null) {
				throw new Exception("The file format couldn't recognize!!");
			}
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

	}

	// Generate Reader for Frame
	public static class GenerateReaderFrame extends GenerateReader {

		private FrameReader frameReader;

		public GenerateReaderFrame(SampleProperties sampleProperties) throws Exception {
			super(sampleProperties);
		}

		public GenerateReaderFrame(String sampleRaw, FrameBlock sampleFrame) throws Exception {
			super(new SampleProperties(sampleRaw, sampleFrame));
		}

		public FrameReader getReader() throws Exception {

			boolean isMapped = readerMapping != null && readerMapping.isMapped();
			if(!isMapped) {
				throw new Exception("Sample raw data and sample frame don't match !!");
			}
			CustomProperties ffp = readerMapping.getFormatProperties();
			if(ffp == null) {
				throw new Exception("The file format couldn't recognize!!");
			}
			// 2. Generate a Frame Reader:
			if(ffp.getRowPattern().equals(CustomProperties.GRPattern.Regular)) {
				if(ffp.getColPattern().equals(CustomProperties.GRPattern.Regular)) {
					frameReader = new FrameGenerateReader.FrameReaderRowRegularColRegular(ffp);
				}
				else {
					frameReader = new FrameGenerateReader.FrameReaderRowRegularColIrregular(ffp);
				}
			}
			else {
				frameReader = new FrameGenerateReader.FrameReaderRowIrregular(ffp);
			}
			return frameReader;
		}
	}
}
