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

package org.tugraz.sysds.runtime.controlprogram.parfor;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.tugraz.sysds.hops.Hop;
import org.tugraz.sysds.runtime.DMLRuntimeException;
import org.tugraz.sysds.runtime.controlprogram.ParForProgramBlock.PDataPartitionFormat;
import org.tugraz.sysds.runtime.controlprogram.caching.MatrixObject;
import org.tugraz.sysds.runtime.matrix.data.InputInfo;
import org.tugraz.sysds.runtime.matrix.data.MatrixBlock;
import org.tugraz.sysds.runtime.matrix.data.OutputInfo;
import org.tugraz.sysds.runtime.meta.DataCharacteristics;
import org.tugraz.sysds.runtime.meta.MatrixCharacteristics;
import org.tugraz.sysds.runtime.meta.MetaDataFormat;
import org.tugraz.sysds.runtime.util.HDFSTool;


/**
 * This is the base class for all data partitioner. 
 * 
 */
public abstract class DataPartitioner 
{	
	
	protected static final Log LOG = LogFactory.getLog(DataPartitioner.class.getName());
	
	//note: the following value has been empirically determined but might change in the future,
	//MatrixBlockDSM.SPARCITY_TURN_POINT (with 0.4) was too high because we create 3-4 values per nnz and 
	//have some computation overhead for binary cell.
	protected static final double SPARSITY_CELL_THRESHOLD = 0.1d; 
	
	protected static final String NAME_SUFFIX = "_dp";
	
	//instance variables
	protected PDataPartitionFormat _format = null;
	protected int _n = -1; //blocksize if applicable
	protected boolean _allowBinarycell = true;
	
	protected DataPartitioner( PDataPartitionFormat dpf, int n ) {
		_format = dpf;
		_n = n;
	}

	public MatrixObject createPartitionedMatrixObject( MatrixObject in, String fnameNew ) {
		return createPartitionedMatrixObject(in, fnameNew, false);
	}

	public MatrixObject createPartitionedMatrixObject( MatrixObject in, String fnameNew, boolean force ) {
		MatrixObject out = new MatrixObject(in.getValueType(), fnameNew);
		return createPartitionedMatrixObject(in, out, force);
	}
	

	/**
	 * Creates a partitioned matrix object based on the given input matrix object, 
	 * according to the specified split format. The input matrix can be in-memory
	 * or still on HDFS and the partitioned output matrix is written to HDFS. The
	 * created matrix object can be used transparently for obtaining the full matrix
	 * or reading 1 or multiple partitions based on given index ranges. 
	 * 
	 * @param in input matrix object
	 * @param out output matrix object
	 * @param force if false, try to optimize
	 * @return partitioned matrix object
	 */
	public MatrixObject createPartitionedMatrixObject( MatrixObject in, MatrixObject out, boolean force ) {
		//check for naive partitioning
		if( _format == PDataPartitionFormat.NONE )
			return in;
		
		//analyze input matrix object
		MetaDataFormat meta = (MetaDataFormat)in.getMetaData();
		DataCharacteristics dc = meta.getDataCharacteristics();
		InputInfo ii = meta.getInputInfo();
		OutputInfo oi = meta.getOutputInfo();
		long rows = dc.getRows();
		long cols = dc.getCols();
		int blen = dc.getBlocksize();
		long nonZeros = dc.getNonZeros();
		double sparsity = dc.dimsKnown(true) ?
				((double)nonZeros)/(rows*cols) : 1.0;
		
		if( !force ) //try to optimize, if format not forced
		{
			//check lower bound of useful data partitioning
			if( rows < Hop.CPThreshold && cols < Hop.CPThreshold )  //or matrix already fits in mem
			{
				return in;
			}
			
			//check for changing to blockwise representations
			if( _format == PDataPartitionFormat.ROW_WISE && cols < Hop.CPThreshold )
			{
				LOG.debug("Changing format from "+PDataPartitionFormat.ROW_WISE+" to "+PDataPartitionFormat.ROW_BLOCK_WISE+".");
				_format = PDataPartitionFormat.ROW_BLOCK_WISE;
			}
			if( _format == PDataPartitionFormat.COLUMN_WISE && rows < Hop.CPThreshold )
			{
				LOG.debug("Changing format from "+PDataPartitionFormat.COLUMN_WISE+" to "+PDataPartitionFormat.ROW_BLOCK_WISE+".");
				_format = PDataPartitionFormat.COLUMN_BLOCK_WISE;
			}
			//_format = PDataPartitionFormat.ROW_BLOCK_WISE_N;
		}
		
		//check changing to binarycell in case of sparse cols (robustness)
		boolean convertBlock2Cell = false;
		if(    ii == InputInfo.BinaryBlockInputInfo 
			&& _allowBinarycell
			&& _format == PDataPartitionFormat.COLUMN_WISE	
			&& sparsity < SPARSITY_CELL_THRESHOLD )
		{
			LOG.debug("Changing partition outputinfo from binaryblock to binarycell due to sparsity="+sparsity);
			oi = OutputInfo.BinaryCellOutputInfo;
			convertBlock2Cell = true;
		}
				
		//prepare filenames and cleanup if required
		String fnameNew = out.getFileName();
		try{
			HDFSTool.deleteFileIfExistOnHDFS(fnameNew);
		}
		catch(Exception ex){
			throw new DMLRuntimeException( ex );
		}
		
		//core partitioning (depending on subclass)
		partitionMatrix( in, fnameNew, ii, oi, rows, cols, blen );
		
		//create output matrix object
		out.setPartitioned( _format, _n ); 
		
		MatrixCharacteristics mcNew = new MatrixCharacteristics( rows, cols, (int)blen );
		mcNew.setNonZeros( nonZeros );
		if( convertBlock2Cell )
			ii = InputInfo.BinaryCellInputInfo;
		MetaDataFormat metaNew = new MetaDataFormat(mcNew,oi,ii);
		out.setMetaData(metaNew);	 
		
		return out;
		
	}

	public void disableBinaryCell()
	{
		_allowBinarycell = false;
	}

	protected abstract void partitionMatrix( MatrixObject in, String fnameNew, InputInfo ii, OutputInfo oi, long rlen, long clen, int blen );

	
	public static MatrixBlock createReuseMatrixBlock( PDataPartitionFormat dpf, int rows, int cols ) 
	{
		MatrixBlock tmp = null;
		
		switch( dpf )
		{
			case ROW_WISE:
				//default assumption sparse, but reset per input block anyway
				tmp = new MatrixBlock( 1, (int)cols, true, (int)(cols*0.1) );
				break;
			case COLUMN_WISE:
				//default dense because single column alwyas below SKINNY_MATRIX_TURN_POINT
				tmp = new MatrixBlock( (int)rows, 1, false );
				break;
			default:
				//do nothing
		}
		
		return tmp;
	}
}
