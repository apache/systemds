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
import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.io.FileFormatProperties;
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Paths;

public class SampleProperties extends FileFormatProperties {

	protected static final Log LOG = LogFactory.getLog(CustomProperties.class.getName());
	private static final long serialVersionUID = 3754623307372402495L;

	private String sampleRaw;
	private MatrixBlock sampleMatrix;
	private Types.DataType dataType;

	public SampleProperties(String sampleRaw, String sampleMatrix, Types.DataType dataType) {

		try {
		this.sampleRaw = new String (Files.readAllBytes( Paths.get(sampleRaw) ));
		sampleMatrix = new String (Files.readAllBytes( Paths.get(sampleMatrix) ));
		this.dataType = dataType;
		String value = null;
		int nrow = 0;
		int ncols = 0;
		InputStream is = IOUtilFunctions.toInputStream(sampleMatrix);
		BufferedReader br = new BufferedReader(new InputStreamReader(is));

			try {
				while((value = br.readLine()) != null) {
					nrow++;
					if(ncols == 0) {
						String[] items = value.split(",");
						ncols = items.length;
					}
				}
				is = IOUtilFunctions.toInputStream(sampleMatrix);
				br = new BufferedReader(new InputStreamReader(is));
				this.sampleMatrix = new MatrixBlock(nrow, ncols, false);
				int row = 0;
				while((value = br.readLine()) != null) { //foreach line
					String cellStr = value.toString().trim();
					String[] parts = IOUtilFunctions.split(cellStr, ",");
					int col = 0;
					for(String part : parts) { //foreach cell
						part = part.trim();
						double cellValue = Double.parseDouble(part);
						this.sampleMatrix.setValue(row, col, cellValue);
						col++;
					}
					row++;
				}
			}
			finally {
				IOUtilFunctions.closeSilently(br);
			}
		}
		catch(Exception ex) {
			throw new RuntimeException("SampleRaw and SampleMatrix Read: " + ex);
		}
	}

	public String getSampleRaw() {
		return sampleRaw;
	}

	public MatrixBlock getSampleMatrix() {
		return sampleMatrix;
	}

	public Types.DataType getDataType() {
		return dataType;
	}
}
