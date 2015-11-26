/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package com.ibm.bi.dml.runtime.instructions.spark.data;

import org.apache.spark.Partitioner;

import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;

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
	
	public BlockPartitioner(MatrixCharacteristics mc, int numParts) 
	{
		long nrblks = mc.getNumRowBlocks();
		long ncblks = mc.getNumColBlocks();
		long nblks = nrblks * ncblks;
		long nblksPerPart = (long)Math.ceil((double)nblks / numParts); 
		long dimBlks = (long) Math.ceil(Math.sqrt(nblksPerPart));
		
		if( nrblks < dimBlks ) { //short and fat
			_rbPerPart = nrblks;
			_cbPerPart = (long)Math.ceil((double)nblksPerPart/_rbPerPart);
		}
		else if( ncblks < dimBlks ) { //tall and skinny
			_cbPerPart = ncblks;
			_rbPerPart = (long)Math.ceil((double)nblksPerPart/_cbPerPart);
		}
		else { //general case
			_rbPerPart = dimBlks;
			_cbPerPart = dimBlks; 
		}
		
		_ncparts = (int)(ncblks/_cbPerPart);
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
		return ixr * _ncparts + ixc;
	}

	@Override
	public int numPartitions() {
		return _numParts;
	}
}
