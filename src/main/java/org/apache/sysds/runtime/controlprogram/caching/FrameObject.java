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

package org.apache.sysds.runtime.controlprogram.caching;


import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.mutable.MutableBoolean;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.parser.DataExpression;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRange;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse;
import org.apache.sysds.runtime.controlprogram.federated.FederationMap;
import org.apache.sysds.runtime.controlprogram.parfor.LocalTaskQueue;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.instructions.spark.data.RDDObject;
import org.apache.sysds.runtime.io.FileFormatProperties;
import org.apache.sysds.runtime.io.FrameReaderFactory;
import org.apache.sysds.runtime.io.FrameWriter;
import org.apache.sysds.runtime.io.FrameWriterFactory;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.lineage.LineageRecomputeUtils;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.meta.MetaData;
import org.apache.sysds.runtime.meta.MetaDataFormat;
import org.apache.sysds.runtime.util.UtilFunctions;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Future;


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
		
		MetaDataFormat metaOld = (MetaDataFormat) fo.getMetaData();
		_metaData = new MetaDataFormat(
			new MatrixCharacteristics(metaOld.getDataCharacteristics()),
			metaOld.getFileFormat());
		_schema = fo._schema.clone();
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
		return ArrayUtils.addAll(
			(_schema!=null) ? _schema : UtilFunctions.nCopies((int)getNumColumns(), ValueType.STRING), 
			(fo._schema!=null) ? fo._schema : UtilFunctions.nCopies((int)fo.getNumColumns(), ValueType.STRING));
	} 
	
	public void setSchema(String schema) {
		if( schema.equals("*") ) {
			//populate default schema
			int clen = (int) getNumColumns();
			if( clen >= 0 ) //known number of cols
				_schema = UtilFunctions.nCopies(clen, ValueType.STRING);
		}
		else 
			_schema = parseSchema(schema);
	}

	public static ValueType[] parseSchema(String schema) {
		if(schema == null)
			return new ValueType[]{ValueType.STRING};
		// parse given schema
		String[] parts = schema.split(DataExpression.DEFAULT_DELIM_DELIMITER);
		ValueType[] ret = new ValueType[parts.length];
		for(int i = 0; i < parts.length; i++)
			ret[i] = ValueType.fromExternalString(parts[i].toUpperCase());
		return ret;
	}
	
	public void setSchema(ValueType[] schema) {
		_schema = schema;
	}
		
	@Override
	public void refreshMetaData() {
		if ( _data == null || _metaData ==null ) //refresh only for existing data
			throw new DMLRuntimeException("Cannot refresh meta data because there is no data or meta data. "); 

		//update matrix characteristics
		DataCharacteristics dc = _metaData.getDataCharacteristics();
		dc.setDimension( _data.getNumRows(),_data.getNumColumns() );
		dc.setNonZeros(_data.getNumRows()*_data.getNumColumns());
		
		//update schema information
		_schema = _data.getSchema();
	}

	public long getNumRows() {
		DataCharacteristics dc = getDataCharacteristics();
		return dc.getRows();
	}

	public long getNumColumns() {
		DataCharacteristics dc = getDataCharacteristics();
		return dc.getCols();
	}
	
	@Override
	protected FrameBlock readBlobFromCache(String fname) throws IOException {
		FrameBlock fb = null;
		if (OptimizerUtils.isUMMEnabled())
			fb = (FrameBlock) UnifiedMemoryManager.readBlock(fname, false);
		else
			fb = (FrameBlock)LazyWriteBuffer.readBlock(fname, false);
		return fb;
	}

	@Override
	protected FrameBlock readBlobFromHDFS(String fname, long[] dims) throws IOException {
		long clen = dims[1];
		MetaDataFormat iimd = (MetaDataFormat) _metaData;
		DataCharacteristics dc = iimd.getDataCharacteristics();

		// handle missing schema if necessary
		ValueType[] lschema = (_schema != null) ? _schema : UtilFunctions.nCopies(clen >= 1 ? (int) clen : 1,
			ValueType.STRING);

		// read the frame block
		FrameBlock data = isFederated() ? acquireReadAndRelease() : FrameReaderFactory
			.createFrameReader(iimd.getFileFormat(), getFileFormatProperties())
			.readFrameFromHDFS(fname, lschema, dc.getRows(), dc.getCols());

		if(iimd.getFileFormat() == FileFormat.CSV)
			_metaData = _metaData instanceof MetaDataFormat ? new MetaDataFormat(data.getDataCharacteristics(),
				iimd.getFileFormat()) : new MetaData(data.getDataCharacteristics());

		// sanity check correct output
		if(data == null)
			throw new IOException("Unable to load frame from file: " + fname);
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
		
		MetaDataFormat iimd = (MetaDataFormat) _metaData;
		DataCharacteristics dc = iimd.getDataCharacteristics();
		int rlen = (int)dc.getRows();
		int clen = (int)dc.getCols();
		
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
		if( fb == null )
			throw new IOException("Unable to load frame from rdd.");
		
		return fb;
	}
	
	@Override
	protected FrameBlock readBlobFromFederated(FederationMap fedMap, long[] dims)
		throws IOException
	{
		FrameBlock ret = new FrameBlock(_schema);
		// provide long support?
		ret.ensureAllocatedColumns((int) dims[0]);
		List<Pair<FederatedRange, Future<FederatedResponse>>> readResponses = fedMap.requestFederatedData();
		try {
			for(Pair<FederatedRange, Future<FederatedResponse>> readResponse : readResponses) {
				FederatedRange range = readResponse.getLeft();
				FederatedResponse response = readResponse.getRight().get();
				// add result
				FrameBlock multRes = (FrameBlock) response.getData()[0];
				for (int r = 0; r < multRes.getNumRows(); r++) {
					for (int c = 0; c < multRes.getNumColumns(); c++) {
						int destRow = range.getBeginDimsInt()[0] + r;
						int destCol = range.getBeginDimsInt()[1] + c;
						ret.set(destRow, destCol, multRes.get(r, c));
					}
				}
			}
		}
		catch(Exception e) {
			throw new DMLRuntimeException("Federated Frame read failed.", e);
		}
		
		return ret;
	}

	@Override
	protected void writeBlobToHDFS(String fname, String ofmt, int rep, FileFormatProperties fprop)
		throws IOException, DMLRuntimeException 
	{
		MetaDataFormat iimd = (MetaDataFormat) _metaData;
		FileFormat fmt = (ofmt != null ? FileFormat.safeValueOf(ofmt) : iimd.getFileFormat());
		
		FrameWriter writer = FrameWriterFactory.createFrameWriter(fmt, fprop);
		writer.writeFrameToHDFS(_data, fname, getNumRows(), getNumColumns());
	}

	@Override
	protected void writeBlobFromRDDtoHDFS(RDDObject rdd, String fname, String ofmt)
		throws IOException, DMLRuntimeException 
	{
		//prepare output info
		MetaDataFormat iimd = (MetaDataFormat) _metaData;

		//note: the write of an RDD to HDFS might trigger
		//lazy evaluation of pending transformations.
		SparkExecutionContext.writeFrameRDDtoHDFS(rdd, fname, iimd.getFileFormat());
	}

	@Override
	protected FrameBlock readBlobFromStream(LocalTaskQueue<IndexedMatrixValue> stream) throws IOException {
		// TODO Auto-generated method stub
		return null;
	}
	
	@Override
	protected FrameBlock reconstructByLineage(LineageItem li) throws IOException {
		return ((FrameObject) LineageRecomputeUtils
			.parseNComputeLineageTrace(li.getData()))
			.acquireReadAndRelease();
	}
}
