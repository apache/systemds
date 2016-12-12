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
import java.util.Arrays;

import org.apache.commons.lang.ArrayUtils;
import org.apache.commons.lang.mutable.MutableBoolean;
import org.apache.sysml.parser.DataExpression;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysml.runtime.instructions.spark.data.RDDObject;
import org.apache.sysml.runtime.io.FrameReader;
import org.apache.sysml.runtime.io.FrameReaderFactory;
import org.apache.sysml.runtime.io.FrameWriter;
import org.apache.sysml.runtime.io.FrameWriterFactory;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.MatrixDimensionsMetaData;
import org.apache.sysml.runtime.matrix.MatrixFormatMetaData;
import org.apache.sysml.runtime.matrix.MetaData;
import org.apache.sysml.runtime.matrix.data.FileFormatProperties;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.runtime.util.UtilFunctions;

public class FrameObject extends CacheableData<FrameBlock>
{
	private static final long serialVersionUID = 1755082174281927785L;

	private ValueType[] _schema = null;

	protected FrameObject() {
		super(DataType.FRAME, ValueType.STRING);
	}

	public FrameObject(String fname) {
		this();
		setFileName(fname);
	}

	public FrameObject(String fname, MetaData meta) {
		this();
		setFileName(fname);
		setMetaData(meta);
	}

	public FrameObject(String fname, MetaData meta, ValueType[] schema) {
		this();
		setFileName(fname);
		setMetaData(meta);
		setSchema(schema);
	}
	
	/**
	 * Copy constructor that copies meta data but NO data.
	 * 
	 * @param fo frame object
	 */
	public FrameObject(FrameObject fo) {
		super(fo);
	}
	
	@Override
	public ValueType[] getSchema() {
		return _schema;
	}

	/**
	 * Obtain schema of value types
	 * 
	 * @param cl column lower bound, inclusive
	 * @param cu column upper bound, inclusive
	 * @return schema of value types
	 */
	public ValueType[] getSchema(int cl, int cu) {
		return (_schema!=null && _schema.length>cu) ? Arrays.copyOfRange(_schema, cl, cu+1) :
			UtilFunctions.nCopies(cu-cl+1, ValueType.STRING);
	}

	/**
	 * Creates a new collection which contains the schema of the current
	 * frame object concatenated with the schema of the passed frame object.
	 * 
	 * @param fo frame object
	 * @return schema of value types
	 */
	public ValueType[] mergeSchemas(FrameObject fo) {
		return (ValueType[]) ArrayUtils.addAll(
			(_schema!=null) ? _schema : UtilFunctions.nCopies((int)getNumColumns(), ValueType.STRING), 
			(fo._schema!=null) ? fo._schema : UtilFunctions.nCopies((int)fo.getNumColumns(), ValueType.STRING));
	} 
	
	public void setSchema(String schema) {
		if( schema.equals("*") ) {
			//populate default schema
			int clen = (int) getNumColumns();
			if( clen > 0 ) //known number of cols
				_schema = UtilFunctions.nCopies(clen, ValueType.STRING);
		}
		else {
			//parse given schema
			String[] parts = schema.split(DataExpression.DEFAULT_DELIM_DELIMITER);
			_schema = new ValueType[parts.length];
			for( int i=0; i<parts.length; i++ )
				_schema[i] = ValueType.valueOf(parts[i].toUpperCase());
		}
	}
	
	public void setSchema(ValueType[] schema) {
		_schema = schema;
	}
		
	@Override
	public void refreshMetaData() 
		throws CacheException
	{
		if ( _data == null || _metaData ==null ) //refresh only for existing data
			throw new CacheException("Cannot refresh meta data because there is no data or meta data. "); 

		//update matrix characteristics
		MatrixCharacteristics mc = ((MatrixDimensionsMetaData) _metaData).getMatrixCharacteristics();
		mc.setDimension( _data.getNumRows(),_data.getNumColumns() );
		mc.setNonZeros(_data.getNumRows()*_data.getNumColumns());
		
		//update schema information
		_schema = _data.getSchema();
	}

	public long getNumRows() {
		MatrixCharacteristics mc = getMatrixCharacteristics();
		return mc.getRows();
	}

	public long getNumColumns() {
		MatrixCharacteristics mc = getMatrixCharacteristics();
		return mc.getCols();
	}
	
	@Override
	protected FrameBlock readBlobFromCache(String fname) throws IOException {
		return (FrameBlock)LazyWriteBuffer.readBlock(fname, false);
	}

	@Override
	protected FrameBlock readBlobFromHDFS(String fname, long rlen, long clen)
		throws IOException 
	{
		MatrixFormatMetaData iimd = (MatrixFormatMetaData) _metaData;
		MatrixCharacteristics mc = iimd.getMatrixCharacteristics();
		
		//handle missing schema if necessary
		ValueType[] lschema = (_schema!=null) ? _schema : 
			UtilFunctions.nCopies(clen>=1 ? (int)clen : 1, ValueType.STRING);
		
		//read the frame block
		FrameBlock data = null;
		try {
			FrameReader reader = FrameReaderFactory.createFrameReader(iimd.getInputInfo(), getFileFormatProperties());
			data = reader.readFrameFromHDFS(fname, lschema, mc.getRows(), mc.getCols()); 
		}
		catch( DMLRuntimeException ex ) {
			throw new IOException(ex);
		}
			
		//sanity check correct output
		if( data == null )
			throw new IOException("Unable to load frame from file: "+fname);
						
		return data;
	}

	@Override
	protected FrameBlock readBlobFromRDD(RDDObject rdd, MutableBoolean status)
			throws IOException 
	{
		//note: the read of a frame block from an RDD might trigger
		//lazy evaluation of pending transformations.
		RDDObject lrdd = rdd;

		//prepare return status (by default only collect)
		status.setValue(false);
		
		MatrixFormatMetaData iimd = (MatrixFormatMetaData) _metaData;
		MatrixCharacteristics mc = iimd.getMatrixCharacteristics();
		int rlen = (int)mc.getRows();
		int clen = (int)mc.getCols();
		
		//handle missing schema if necessary
		ValueType[] lschema = (_schema!=null) ? _schema : 
			UtilFunctions.nCopies(clen>=1 ? (int)clen : 1, ValueType.STRING);
		
		FrameBlock fb = null;
		try  {
			//prevent unnecessary collect through rdd checkpoint
			if( rdd.allowsShortCircuitCollect() ) {
				lrdd = (RDDObject)rdd.getLineageChilds().get(0);
			}
			
			//collect frame block from binary block RDD
			fb = SparkExecutionContext.toFrameBlock(lrdd, lschema, rlen, clen);	
		}
		catch(DMLRuntimeException ex) {
			throw new IOException(ex);
		}
				
		//sanity check correct output
		if( fb == null ) {
			throw new IOException("Unable to load frame from rdd: "+lrdd.getVarName());
		}
		
		return fb;
	}

	@Override
	protected void writeBlobToHDFS(String fname, String ofmt, int rep, FileFormatProperties fprop) 
		throws IOException, DMLRuntimeException 
	{
		OutputInfo oinfo = OutputInfo.stringToOutputInfo(ofmt);
		FrameWriter writer = FrameWriterFactory.createFrameWriter(oinfo, fprop);
		writer.writeFrameToHDFS(_data, fname,  getNumRows(), getNumColumns());
	}

	@Override
	protected void writeBlobFromRDDtoHDFS(RDDObject rdd, String fname, String ofmt) 
		throws IOException, DMLRuntimeException 
	{
		//prepare output info
		MatrixFormatMetaData iimd = (MatrixFormatMetaData) _metaData;
		OutputInfo oinfo = (ofmt != null ? OutputInfo.stringToOutputInfo (ofmt ) 
				: InputInfo.getMatchingOutputInfo (iimd.getInputInfo ()));
	    
		//note: the write of an RDD to HDFS might trigger
		//lazy evaluation of pending transformations.				
		SparkExecutionContext.writeFrameRDDtoHDFS(rdd, fname, oinfo);	
	}

}
