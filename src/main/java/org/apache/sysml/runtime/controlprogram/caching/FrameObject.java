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

package org.apache.sysml.runtime.controlprogram.caching;

import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;

public class FrameObject extends CacheableData
{
	private static final long serialVersionUID = 1755082174281927785L;

	/** Container object that holds the actual data.*/
	private FrameBlock _data = null;

	/** The name of HDFS file in which the data is backed up. */
	private String _hdfsFileName = null; // file name and path

	protected FrameObject() {
		super(DataType.FRAME, ValueType.UNKNOWN);
	}
	
	public FrameObject(String fname, FrameBlock data) {
		this();
		setFileName(fname);
		setData(data);
	}
	
	public void setFileName(String fname) {
		_hdfsFileName = fname;
	}
	
	public String getFileName() {
		return _hdfsFileName;
	}
	
	/**
	 * NOTE: temporary API until integrated into caching.
	 * 
	 * @param block
	 */
	public void setData(FrameBlock data) {
		_data = data;
	}
	
	/**
	 * NOTE: temporary API until integrated into caching.
	 * 
	 * @return
	 */
	public FrameBlock getData() {
		return _data;
	}
	
	
	////////////////////////////////////
	// currently unsupported caching api
	
	@Override
	protected boolean isBlobPresent() {
		throw new RuntimeException("Caching not implemented yet for FrameObject.");
	}

	@Override
	protected void evictBlobFromMemory(MatrixBlock mb) throws CacheIOException {
		//TODO refactoring api (no dependence on matrixblock)
		throw new RuntimeException("Caching not implemented yet for FrameObject.");
	}

	@Override
	protected void restoreBlobIntoMemory() throws CacheIOException {
		throw new RuntimeException("Caching not implemented yet for FrameObject.");
	}

	@Override
	protected void freeEvictedBlob() {
		throw new RuntimeException("Caching not implemented yet for FrameObject.");
	}

	@Override
	protected boolean isBelowCachingThreshold() {
		throw new RuntimeException("Caching not implemented yet for FrameObject.");
	}

	@Override
	public String getDebugName() {
		throw new RuntimeException("Caching not implemented yet for FrameObject.");
	}
}
