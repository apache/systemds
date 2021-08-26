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

public class SampleProperties extends FileFormatProperties {

	protected static final Log LOG = LogFactory.getLog(CustomProperties.class.getName());
	private static final long serialVersionUID = 3754623307372402495L;
	private String sampleRawFileName;
	private String sampleBinaryFileName;
	private int sampleRows;
	private int sampleCols;


	private Types.DataType dataType;

	public SampleProperties(String sampleRawFileName, String sampleBinaryFileName, int sampleRows, int sampleCols, Types.DataType dataType) {
		this.sampleRawFileName = sampleRawFileName;
		this.sampleBinaryFileName = sampleBinaryFileName;
		this.sampleRows = sampleRows;
		this.sampleCols = sampleCols;
		this.dataType = dataType;
	}

	public String getSampleRawFileName() {
		return sampleRawFileName;
	}

	public void setSampleRawFileName(String sampleRawFileName) {
		this.sampleRawFileName = sampleRawFileName;
	}

	public String getSampleBinaryFileName() {
		return sampleBinaryFileName;
	}

	public void setSampleBinaryFileName(String sampleBinaryFileName) {
		this.sampleBinaryFileName = sampleBinaryFileName;
	}

	public Types.DataType getDataType() {
		return dataType;
	}

	public int getSampleRows() {
		return sampleRows;
	}

	public int getSampleCols() {
		return sampleCols;
	}
}
