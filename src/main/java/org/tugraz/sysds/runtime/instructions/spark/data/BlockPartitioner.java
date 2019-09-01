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

package org.tugraz.sysds.runtime.instructions.spark.data;

import org.apache.spark.Partitioner;
import org.tugraz.sysds.runtime.matrix.data.MatrixIndexes;
import org.tugraz.sysds.runtime.meta.DataCharacteristics;

import java.util.Arrays;

/**
 * Default partitioner used for all binary block rdd operations in order
 * to enable sufficient local aggregation independent of the aggregation
 * direction (row/col-wise). Especially, on large squared matrices 
 * (as common for factorization or graph algorithms), this is crucial 
 * for performance. 
 * 
 */
public class BlockPartitioner extends Partitioner
{
	private static final long serialVersionUID = 3207938407732880324L;
	
	private int _numParts = -1;
	private int _ncparts = -1;
	private long _rbPerPart = -1;
	private long _cbPerPart = -1;
	
	public BlockPartitioner(DataCharacteristics mc, int numParts)
	{
		//sanity check known dimensions
		if( !mc.dimsKnown() || mc.getBlocksize()<1 || mc.getBlocksize()<1 ) {
			throw new RuntimeException("Invalid unknown matrix characteristics.");
		}
		
		//prepare meta data
		long nrblks = mc.getNumRowBlocks();
		long ncblks = mc.getNumColBlocks();
		long nblks = nrblks * ncblks;
		
		//compute perfect squared tile-size (via flooring to
		//avoid empty partitions; overflow handled via mod numParts)
		double nblksPerPart = Math.max((double)nblks/numParts,1);
		long dimBlks = (long)Math.max(Math.floor(Math.sqrt(nblksPerPart)),1);
		
		//adjust tile shape according to matrix shape
		if( nrblks < dimBlks ) { //short and fat
			_rbPerPart = nrblks;
			_cbPerPart = (long)Math.max(Math.floor(nblksPerPart/_rbPerPart),1);
		}
		else if( ncblks < dimBlks ) { //tall and skinny
			_cbPerPart = ncblks;
			_rbPerPart = (long)Math.max(Math.floor(nblksPerPart/_cbPerPart),1);
		}
		else { //general case
			_rbPerPart = dimBlks;
			_cbPerPart = dimBlks; 
		}
		
		//compute meta data for runtime
		_ncparts = (int)Math.ceil((double)ncblks/_cbPerPart);
		_numParts = numParts;
	}
	
	@Override
	public int getPartition(Object arg0) 
	{
		//sanity check for valid class
		if( !(arg0 instanceof MatrixIndexes) ) {
			throw new RuntimeException("Unsupported key class "
					+ "(expected MatrixIndexes): "+arg0.getClass().getName());
		}
			
		//get partition id
		MatrixIndexes ix = (MatrixIndexes) arg0;
		int ixr = (int)((ix.getRowIndex()-1)/_rbPerPart);
		int ixc = (int)((ix.getColumnIndex()-1)/_cbPerPart);
		int id = ixr * _ncparts + ixc;
		
		//ensure valid range
		return id % _numParts;
	}

	@Override
	public int numPartitions() {
		return _numParts;
	}
	
	@Override 
	public int hashCode() {
		return Arrays.hashCode(new long[]{
			_numParts, _ncparts, _rbPerPart, _cbPerPart});
	}

	@Override
	public boolean equals(Object obj) 
	{
		if( !(obj instanceof BlockPartitioner) )
			return false;
		
		BlockPartitioner that = (BlockPartitioner) obj;
		return _numParts == that._numParts
			&& _ncparts == that._ncparts
			&& _rbPerPart == that._rbPerPart
			&& _cbPerPart == that._cbPerPart;
	}
}
