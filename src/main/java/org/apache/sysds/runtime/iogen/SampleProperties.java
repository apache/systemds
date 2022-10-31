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
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.io.FileFormatProperties;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public class SampleProperties extends FileFormatProperties {

	protected static final Log LOG = LogFactory.getLog(CustomProperties.class.getName());

	private String sampleRaw;
	private MatrixBlock sampleMatrix;
	private FrameBlock sampleFrame;
	private Types.DataType dataType;

	public SampleProperties(String sampleRaw) {
		this.sampleRaw = sampleRaw;
	}

	public SampleProperties(String sampleRaw, MatrixBlock sampleMatrix) {
		this.sampleRaw = sampleRaw;
		this.sampleMatrix = sampleMatrix;
		this.dataType = Types.DataType.MATRIX;
	}

	public SampleProperties(String sampleRaw, FrameBlock sampleFrame) {
		this.sampleRaw = sampleRaw;
		this.sampleFrame = sampleFrame;
		this.dataType = Types.DataType.FRAME;
	}

	public String getSampleRaw() {
		return sampleRaw;
	}

	public MatrixBlock getSampleMatrix() {
		return sampleMatrix;
	}

	public FrameBlock getSampleFrame() {
		return sampleFrame;
	}

	public Types.DataType getDataType() {
		return dataType;
	}

	public void setSampleMatrix(MatrixBlock sampleMatrix) {
		this.sampleMatrix = sampleMatrix;
		dataType = Types.DataType.MATRIX;
	}

	public void setSampleFrame(FrameBlock sampleFrame) {
		this.sampleFrame = sampleFrame;
		dataType = Types.DataType.FRAME;
	}
}
