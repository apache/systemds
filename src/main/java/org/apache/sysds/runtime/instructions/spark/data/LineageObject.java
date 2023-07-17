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

package org.apache.sysds.runtime.instructions.spark.data;

import java.util.ArrayList;
import java.util.List;

import org.apache.sysds.runtime.controlprogram.caching.CacheableData;

public abstract class LineageObject 
{
	//basic lineage information
	protected int _numRef = -1;
	protected int _maxNumRef = -1;
	protected boolean _lineageCached = false;
	protected final List<LineageObject> _childs;
	
	//N:1 back reference to matrix/frame object
	protected CacheableData<?> _cd = null;
	
	protected LineageObject() {
		_numRef = 0;
		_lineageCached = false;
		_childs = new ArrayList<>();
	}
	
	public int getNumReferences() {
		return _numRef;
	}
	
	public void setBackReference(CacheableData<?> cd) {
		_cd = cd;
	}
	
	public boolean hasBackReference() {
		return (_cd != null);
	}

	public void setLineageCached() {
		_lineageCached = true;
	}

	public boolean isInLineageCache() {
		return _lineageCached;
	}
	
	public void incrementNumReferences() {
		_numRef++;

		// Maintain the maximum reference count. Higher reference
		// count indicates higher importance to persist (in lineage cache)
		_maxNumRef = Math.max(_numRef, _maxNumRef);
	}
	
	public void decrementNumReferences() {
		_numRef--;
	}

	public int getMaxReferenceCount() {
		return _maxNumRef;
	}
	
	public List<LineageObject> getLineageChilds() {
		return _childs;
	}
	
	public void addLineageChild(LineageObject lob) {
		lob.incrementNumReferences();
		_childs.add( lob );
	}

	public void removeAllChild() {
		_childs.clear();
	}
}
