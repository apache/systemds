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

package org.apache.sysml.runtime.instructions.flink.data;

import org.apache.flink.api.java.DataSet;
import org.apache.sysml.runtime.instructions.spark.data.LineageObject;

public class DataSetObject extends LineageObject {

	private DataSet<?> _dsHandle = null;

	//meta data on origin of given dataset handle
	private boolean _checkpointed = false; //created via checkpoint instruction
	private boolean _hdfsfile = false;     //created from hdfs file
	private String _hdfsFname = null;     //hdfs filename, if created from hdfs.

	public DataSetObject(DataSet<?> dsvar, String varName) {
		_dsHandle = dsvar;
		_varName = varName;
	}

	public DataSet<?> getDataSet() {
		return _dsHandle;
	}

	public boolean isCheckpointed() {
		return _checkpointed;
	}

	public void setCheckpointed(boolean checkpointed) {
		this._checkpointed = checkpointed;
	}

	public boolean isHDFSFile() {
		return _hdfsfile;
	}

	public void setHDFSFile(boolean hdfsfile) {
		this._hdfsfile = hdfsfile;
	}

	public String getHDFSFilename() {
		return _hdfsFname;
	}

	public void setHDFSFilename(String fname) {
		this._hdfsFname = fname;
	}


}
