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

package org.apache.sysds.runtime.io;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import java.io.Serializable;

public class FileFormatPropertiesHDF5 extends FileFormatProperties implements Serializable {
	protected static final Log LOG = LogFactory.getLog(FileFormatPropertiesHDF5.class.getName());
	private static final long serialVersionUID = 8646275033790103030L;

	private String datasetName;

	public FileFormatPropertiesHDF5() {
		this.datasetName = "systemdsh5";
	}

	public FileFormatPropertiesHDF5(String datasetName) {
		this.datasetName = datasetName;
	}

	public String getDatasetName() {
		return datasetName;
	}

	@Override public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(" datasetName " + datasetName);
		return sb.toString();
	}
}
