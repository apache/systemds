/*
 * Modifications Copyright 2019 Graz University of Technology
 *
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

package org.tugraz.sysds.runtime.controlprogram.caching;


import org.apache.commons.lang.mutable.MutableBoolean;
import org.apache.spark.api.java.JavaPairRDD;
import org.tugraz.sysds.common.Types.DataType;
import org.tugraz.sysds.common.Types.ValueType;
import org.tugraz.sysds.runtime.DMLRuntimeException;
import org.tugraz.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.tugraz.sysds.runtime.data.HomogTensor;
import org.tugraz.sysds.runtime.data.TensorIndexes;
import org.tugraz.sysds.runtime.instructions.spark.data.RDDObject;
import org.tugraz.sysds.runtime.io.FileFormatProperties;
import org.tugraz.sysds.runtime.meta.DataCharacteristics;
import org.tugraz.sysds.runtime.meta.MetaData;
import org.tugraz.sysds.runtime.meta.TensorCharacteristics;

import java.io.IOException;

public class TensorObject extends CacheableData<HomogTensor>
{
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
		if ( _data == null || _metaData ==null ) //refresh only for existing data
			throw new DMLRuntimeException("Cannot refresh meta data because there is no data or meta data. "); 

		//update matrix characteristics
		DataCharacteristics tc = _metaData.getDataCharacteristics();
		long[] dims = new long[_data.getNumDims()];
		for (int i = 0; i < _data.getNumDims(); i++) {
			dims[i] = _data.getDim(i);
		}
		tc.setDims(dims);
		tc.setNonZeros(_data.getNonZeros());
	}

	public int[] getDims() {
		return _data.getDims();
	}

	public long getNumRows() {
		DataCharacteristics dc = getDataCharacteristics();
		return dc.getRows();
	}

	public long getNumColumns() {
		DataCharacteristics dc = getDataCharacteristics();
		return dc.getCols();
	}

	public long getNnz() {
		return getDataCharacteristics().getNonZeros();
	}

	@Override
	protected HomogTensor readBlobFromCache(String fname) throws IOException {
		return (HomogTensor)LazyWriteBuffer.readBlock(fname, false);
	}

	@Override
	protected HomogTensor readBlobFromHDFS(String fname, long rlen, long clen)
		throws IOException 
	{
		//TODO read from HDFS
		return null;
	}

	@Override
	@SuppressWarnings("unchecked")
	protected HomogTensor readBlobFromRDD(RDDObject rdd, MutableBoolean status)
		throws IOException
	{
		status.setValue(false);
		TensorCharacteristics tc = (TensorCharacteristics) _metaData.getDataCharacteristics();
		// TODO correct blocksize;
		// TODO read from RDD
		return SparkExecutionContext.toTensorBlock((JavaPairRDD<TensorIndexes, TensorBlock>)rdd.getRDD(), tc);
	}

	@Override
	protected void writeBlobToHDFS(String fname, String ofmt, int rep, FileFormatProperties fprop) 
		throws IOException, DMLRuntimeException 
	{
		//TODO write
	}

	@Override
	protected void writeBlobFromRDDtoHDFS(RDDObject rdd, String fname, String ofmt) 
		throws IOException, DMLRuntimeException 
	{
		//TODO rdd write
	}
}
