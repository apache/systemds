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

package org.apache.sysml.runtime.instructions.spark.data;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

public class DatasetObject extends LineageObject
{
	private final Dataset<Row> _dsHandle;
	private final boolean _isVector;
	private final boolean _containsID;
	
	public DatasetObject( Dataset<Row> dsvar, String varName) {
		this(dsvar, varName, true, true);
	}
	
	public DatasetObject( Dataset<Row> dsvar, String varName, boolean isVector, boolean containsID) {
		super(varName);
		_dsHandle = dsvar;
		_isVector = isVector;
		_containsID = containsID;
	}

	public Dataset<Row> getDataset() {
		return _dsHandle;
	}
	
	public boolean isVectorBased() {
		return _isVector;
	}
	
	public boolean containsID() {
		return _containsID;
	}
}
