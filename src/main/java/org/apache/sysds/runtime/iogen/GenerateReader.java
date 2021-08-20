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

import org.apache.sysds.runtime.io.*;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public class GenerateReader {

	/*
	Generate Reader has two steps:
		1. Identify file format and extract the properties of it based on the Sample Matrix.
		 The ReaderMapping class tries to map the Sample Matrix on the Sample Raw Matrix.
		 The result of a ReaderMapping is a FileFormatProperties object.

		2. Generate a reader based on inferred properties.
	 */
	public static MatrixReader generateReader(String sampleRaw, MatrixBlock sampleMatrix) throws Exception {

		// 1. Identify file format properties:
		ReaderMapping rp = new ReaderMapping(sampleRaw, sampleMatrix);

		boolean isMapped = rp.isMapped();
		if(!isMapped) {
			throw new Exception("Sample raw data and sample matrix don't match !!");
		}
		FileFormatProperties ffp = rp.getFormatProperties();
		if(ffp == null) {
			throw new Exception("The file format couldn't recognize!!");
		}

		// 2. Generate a Matrix Reader:
		MatrixReader reader = null;
		//TODO: after identify file format properties we have to return a Matrix Reader
		return reader;
	}
}
