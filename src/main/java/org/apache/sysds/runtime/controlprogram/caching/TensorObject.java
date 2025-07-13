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


import org.apache.commons.lang3.mutable.MutableBoolean;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.FederationMap;
import org.apache.sysds.runtime.controlprogram.parfor.LocalTaskQueue;
import org.apache.sysds.runtime.data.TensorBlock;
import org.apache.sysds.runtime.data.TensorIndexes;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.instructions.spark.data.RDDObject;
import org.apache.sysds.runtime.io.FileFormatProperties;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.lineage.LineageRecomputeUtils;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.meta.MetaData;
import org.apache.sysds.runtime.meta.MetaDataFormat;
import org.apache.sysds.runtime.meta.TensorCharacteristics;
import org.apache.sysds.runtime.util.DataConverter;

import java.io.IOException;
import java.util.Arrays;

public class TensorObject extends CacheableData<TensorBlock> {
	private static final long serialVersionUID = -2843358400200380775L;

	protected TensorObject() {
		super(DataType.TENSOR, ValueType.STRING);
	}

	public TensorObject(String fname) {
		this();
		setFileName(fname);
	}

	public TensorObject(ValueType vt, String fname) {
		super(DataType.TENSOR, vt);
		setFileName(fname);
	}
	
	public TensorObject(String fname, MetaData meta) {
		this();
		setFileName(fname);
		setMetaData(meta);
	}

	/**
	 * Copy constructor that copies meta data but NO data.
	 *
	 * @param fo frame object
	 */
	public TensorObject(TensorObject fo) {
		super(fo);
	}

	@Override
	public void refreshMetaData() {
		if ( _data == null || _metaData == null ) //refresh only for existing data
			throw new DMLRuntimeException("Cannot refresh meta data because there is no data or meta data. ");

		//update matrix characteristics
		DataCharacteristics tc = _metaData.getDataCharacteristics();
		long[] dims = _metaData.getDataCharacteristics().getDims();
		tc.setDims(dims);
		tc.setNonZeros(_data.getNonZeros());
	}

	public long getNumRows() {
		DataCharacteristics dc = getDataCharacteristics();
		return dc.getDim(0);
	}

	public long getNumColumns() {
		DataCharacteristics dc = getDataCharacteristics();
		return dc.getDim(1);
	}

	public long getNnz() {
		return getDataCharacteristics().getNonZeros();
	}

	@Override
	protected TensorBlock readBlobFromCache(String fname) throws IOException {
		TensorBlock tb = null;
		if (OptimizerUtils.isUMMEnabled())
			tb = (TensorBlock) UnifiedMemoryManager.readBlock(fname, false);
		else
			tb = (TensorBlock) LazyWriteBuffer.readBlock(fname, false);
		return tb;
	}

	@Override
	protected TensorBlock readBlobFromHDFS(String fname, long[] dims)
			throws IOException {
		MetaDataFormat iimd = (MetaDataFormat) _metaData;
		DataCharacteristics dc = iimd.getDataCharacteristics();
		long begin = 0;

		if( LOG.isTraceEnabled() ) {
			LOG.trace("Reading tensor from HDFS...  " + hashCode() + "  Path: " + fname +
					", dimensions: " + Arrays.toString(dims));
			begin = System.currentTimeMillis();
		}

		int blen = dc.getBlocksize();
		//read tensor and maintain meta data
		TensorBlock newData = DataConverter.readTensorFromHDFS(fname, iimd.getFileFormat(), dims, blen, getSchema());
		setHDFSFileExists(true);

		//sanity check correct output
		if( newData == null )
			throw new IOException("Unable to load tensor from file: "+fname);

		if( LOG.isTraceEnabled() )
			LOG.trace("Reading Completed: " + (System.currentTimeMillis()-begin) + " msec.");

		return newData;
	}

	@Override
	@SuppressWarnings("unchecked")
	protected TensorBlock readBlobFromRDD(RDDObject rdd, MutableBoolean status) {
		status.setValue(false);
		TensorCharacteristics tc = (TensorCharacteristics) _metaData.getDataCharacteristics();
		// TODO correct blocksize;
		// TODO read from RDD
		return SparkExecutionContext.toTensorBlock((JavaPairRDD<TensorIndexes, TensorBlock>) rdd.getRDD(), tc);
	}
	
	@Override
	protected TensorBlock readBlobFromFederated(FederationMap fedMap, long[] dims)
		throws IOException
	{
		throw new DMLRuntimeException("Unsupported federated tensors");
	}

	@Override
	protected void writeBlobToHDFS(String fname, String ofmt, int rep, FileFormatProperties fprop)
			throws IOException, DMLRuntimeException {
		long begin = 0;
		if (LOG.isTraceEnabled()) {
			LOG.trace(" Writing tensor to HDFS...  " + hashCode() + "  Path: " + fname + ", Format: " +
					(ofmt != null ? ofmt : "inferred from metadata"));
			begin = System.currentTimeMillis();
		}

		MetaDataFormat iimd = (MetaDataFormat) _metaData;

		if (_data != null) {
			// Get the dimension information from the metadata stored within TensorObject
			DataCharacteristics dc = iimd.getDataCharacteristics();
			// Write the tensor to HDFS in requested format
			FileFormat fmt = (ofmt != null ? FileFormat.safeValueOf(ofmt) : iimd.getFileFormat());

			//TODO check correct blocking
			DataConverter.writeTensorToHDFS(_data, fname, fmt, dc);
			if( LOG.isTraceEnabled() )
				LOG.trace("Writing tensor to HDFS ("+fname+") - COMPLETED... " + (System.currentTimeMillis()-begin) + " msec.");
		}
		else if (LOG.isTraceEnabled()) {
			LOG.trace("Writing tensor to HDFS (" + fname + ") - NOTHING TO WRITE (_data == null).");
		}
		if( DMLScript.STATISTICS )
			CacheStatistics.incrementHDFSWrites();
	}

	@Override
	protected ValueType[] getSchema() {
		return _data.getSchema();
	}

	@Override
	protected void writeBlobFromRDDtoHDFS(RDDObject rdd, String fname, String ofmt)
			throws DMLRuntimeException {
		//TODO rdd write
	}
	

	@Override
	protected TensorBlock readBlobFromStream(LocalTaskQueue<IndexedMatrixValue> stream) throws IOException {
		// TODO Auto-generated method stub
		return null;
	}
	
	@Override
	protected TensorBlock reconstructByLineage(LineageItem li) throws IOException {
		return ((TensorObject) LineageRecomputeUtils
			.parseNComputeLineageTrace(li.getData()))
			.acquireReadAndRelease();
	}
}
