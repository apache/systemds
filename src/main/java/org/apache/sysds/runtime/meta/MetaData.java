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

package org.apache.sysds.runtime.meta;

/**
 * Class to store metadata associated with a file (e.g., a matrix) on disk.
 *
 */
public class MetaData
{
	protected final DataCharacteristics _dc;
	
	public MetaData() {
		this(new MatrixCharacteristics());
	}
	
	public MetaData(DataCharacteristics dc) {
		_dc = dc;
	}
	
	public DataCharacteristics getDataCharacteristics() {
		return _dc;
	}
	
	@Override
	public boolean equals (Object anObject) {
		if( !(anObject instanceof MetaData) )
			return false;
		MetaData that = (MetaData)anObject;
		return _dc.equals(that._dc);
	}
	
	@Override
	public int hashCode() {
		return _dc.hashCode();
	}

	@Override
	public String toString() {
		return _dc.toString();
	}
	
	@Override
	public Object clone() {
		if (_dc instanceof MatrixCharacteristics)
			return new MetaData(new MatrixCharacteristics(_dc));
		else
			return new MetaData(new TensorCharacteristics(_dc));
	}
}
