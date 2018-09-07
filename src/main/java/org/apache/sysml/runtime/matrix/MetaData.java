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

package org.apache.sysml.runtime.matrix;

/**
 * Class to store metadata associated with a file (e.g., a matrix) on disk.
 *
 */
public class MetaData
{
	protected final MatrixCharacteristics _mc;
	
	public MetaData(MatrixCharacteristics mc) {
		_mc = mc;
	}
	
	public MatrixCharacteristics getMatrixCharacteristics() {
		return _mc;
	}
	
	@Override
	public boolean equals (Object anObject) {
		if( !(anObject instanceof MetaData) )
			return false;
		MetaData that = (MetaData)anObject;
		return _mc.equals(that._mc);
	}
	
	@Override
	public int hashCode() {
		return _mc.hashCode();
	}

	@Override
	public String toString() {
		return _mc.toString();
	}
	
	@Override
	public Object clone() {
		return new MetaData(new MatrixCharacteristics(_mc));
	}
}
