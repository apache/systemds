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
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.MatrixDimensionsMetaData;
import org.apache.sysml.runtime.matrix.MetaData;
import org.apache.sysml.runtime.matrix.data.FileFormatProperties;
import org.apache.sysml.runtime.matrix.data.FrameBlock;

public class FrameObject extends CacheableData<FrameBlock>
{
	private static final long serialVersionUID = 1755082174281927785L;

	/** Holds meta data on frame characteristics (required for nrow etc)*/
	private MetaData _metaData = null;
	
	/**
	 * 
	 */
	protected FrameObject() {
		super(DataType.FRAME, ValueType.UNKNOWN);
	}
	
	/**
	 * 
	 * @param fname
	 */
	public FrameObject(String fname) {
		this();
		setFileName(fname);
	}
	
	/**
	 * 
	 * @param fname
	 * @param meta
	 */
	public FrameObject(String fname, MetaData meta) {
		this();
		setFileName(fname);
		setMetaData(meta);
	}
	
	/**
	 * Copy constructor that copies meta data but NO data.
	 * 
	 * @param fo
	 */
	public FrameObject(FrameObject fo) {
		super(fo);
	}
	
	@Override
	public void setMetaData(MetaData md) {
		_metaData = md;
	}
	
	@Override
	public MetaData getMetaData() {
		return _metaData;
	}

	@Override
	public void removeMetaData() {
		_metaData = null;
	}
	
	public void refreshMetaData() 
		throws CacheException
	{
		if ( _data == null || _metaData ==null ) //refresh only for existing data
			throw new CacheException("Cannot refresh meta data because there is no data or meta data. "); 
		
		MatrixCharacteristics mc = ((MatrixDimensionsMetaData) _metaData).getMatrixCharacteristics();
		mc.setDimension( _data.getNumRows(),_data.getNumColumns() );	
	}
	
	////////////////////////////////////
	// high-level cache API (pseudo-integrated)


	@Override
	public FrameBlock acquireRead() 
		throws CacheException 
	{
		//read in=memory frame if necessary
		if( _data == null ) {
			// TODO @Arvind: please integrate readers here for now
		}
		
		return _data;
	}

	@Override
	public FrameBlock acquireModify() 
		throws CacheException 
	{
		return _data;
	}

	@Override
	public FrameBlock acquireModify(FrameBlock newData) 
		throws CacheException 
	{
		//set data and update meta data
		_data = newData;
		refreshMetaData();
		
		return _data;
	}

	@Override
	public void release() 
		throws CacheException 
	{
		//do nothing
	}

	@Override
	public void clearData() 
		throws CacheException 
	{
		if( isCleanupEnabled() )
			_data = null;
	}

	@Override
	public void exportData(String fName, String outputFormat, int replication, FileFormatProperties formatProperties) 
		throws CacheException 
	{
		// TODO @Arvind: please integrate writers here for now		
	}
	
	////////////////////////////////////
	// currently unsupported caching api
	
	@Override
	protected boolean isBlobPresent() {
		throw new RuntimeException("Caching not implemented yet for FrameObject.");
	}

	@Override
	protected void evictBlobFromMemory(FrameBlock mb) throws CacheIOException {
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
