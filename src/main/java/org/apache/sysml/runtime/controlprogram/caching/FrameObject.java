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


import java.io.IOException;

import org.apache.commons.lang.mutable.MutableBoolean;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.instructions.spark.data.RDDObject;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.MatrixDimensionsMetaData;
import org.apache.sysml.runtime.matrix.MetaData;
import org.apache.sysml.runtime.matrix.data.FileFormatProperties;
import org.apache.sysml.runtime.matrix.data.FrameBlock;

public class FrameObject extends CacheableData<FrameBlock>
{
	private static final long serialVersionUID = 1755082174281927785L;

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
	public void refreshMetaData() 
		throws CacheException
	{
		if ( _data == null || _metaData ==null ) //refresh only for existing data
			throw new CacheException("Cannot refresh meta data because there is no data or meta data. "); 
		
		MatrixCharacteristics mc = ((MatrixDimensionsMetaData) _metaData).getMatrixCharacteristics();
		mc.setDimension( _data.getNumRows(),_data.getNumColumns() );	
	}
	

	@Override
	protected FrameBlock readBlobFromCache(String fname) throws IOException {
		return (FrameBlock)LazyWriteBuffer.readBlock(fname, false);
	}

	@Override
	protected FrameBlock readBlobFromHDFS(String fname, long rlen, long clen)
		throws IOException 
	{
		throw new IOException("Not implemented yet.");
	}

	@Override
	protected FrameBlock readBlobFromRDD(RDDObject rdd, MutableBoolean status)
			throws IOException 
	{
		throw new IOException("Not implemented yet.");
	}

	@Override
	protected void writeBlobToHDFS(String fname, String ofmt, int rep, FileFormatProperties fprop) 
		throws IOException, DMLRuntimeException 
	{
		throw new IOException("Not implemented yet.");
	}

	@Override
	protected void writeBlobFromRDDtoHDFS(RDDObject rdd, String fname, String ofmt) 
		throws IOException, DMLRuntimeException 
	{
		throw new IOException("Not implemented yet.");
	}
}
