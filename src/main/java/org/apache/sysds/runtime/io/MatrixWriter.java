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

package org.apache.sysds.runtime.io;

import java.io.IOException;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

/**
 * Base class for all format-specific matrix writers. Every writer is required to implement the basic 
 * write functionality but might provide additional custom functionality. Any non-default parameters
 * (e.g., CSV read properties) should be passed into custom constructors. There is also a factory
 * for creating format-specific writers. 
 */
public abstract class MatrixWriter {
	protected static final Log LOG = LogFactory.getLog(MatrixWriter.class.getName());
	
	protected boolean _forcedParallel = false;
	
	public void writeMatrixToHDFS( MatrixBlock src, String fname, long rlen, long clen, int blen, long nnz ) throws IOException {
		writeMatrixToHDFS(src, fname, rlen, clen, blen, nnz, false);
	}

	public abstract void writeMatrixToHDFS( MatrixBlock src, String fname, long rlen, long clen, int blen, long nnz, boolean diag )
		throws IOException;
	
	public void setForcedParallel(boolean par) {
		_forcedParallel = par;
	}
	
	
	/**
	 * Writes a minimal entry to represent an empty matrix on hdfs.
	 * 
	 * @param fname file name
	 * @param rlen number of rows
	 * @param clen number of columns
	 * @param blen number of rows/cols in block
	 * @throws IOException if IOException occurs
	 */
	public abstract void writeEmptyMatrixToHDFS(String fname, long rlen, long clen, int blen)
		throws IOException;

	public static MatrixBlock[] createMatrixBlocksForReuse( long rlen, long clen, int blen, boolean sparse, long nonZeros ) {
		MatrixBlock[] blocks = new MatrixBlock[4];
		double sparsity = ((double)nonZeros)/(rlen*clen);
		long estNNZ = -1;
		
		//full block 
		if( rlen >= blen && clen >= blen ) {
			estNNZ = (long) (blen*blen*sparsity);
			blocks[0] = new MatrixBlock( blen, blen, sparse, (int)estNNZ );
		}
		//partial col block
		if( rlen >= blen && clen%blen!=0 ) {
			estNNZ = (long) (blen*(clen%blen)*sparsity);
			blocks[1] = new MatrixBlock( blen, (int)(clen%blen), sparse, (int)estNNZ );
		}
		//partial row block
		if( rlen%blen!=0 && clen>=blen ) {
			estNNZ = (long) ((rlen%blen)*blen*sparsity);
			blocks[2] = new MatrixBlock( (int)(rlen%blen), blen, sparse, (int)estNNZ );
		}
		//partial row/col block
		if( rlen%blen!=0 && clen%blen!=0 ) {
			estNNZ = (long) ((rlen%blen)*(clen%blen)*sparsity);
			blocks[3] = new MatrixBlock( (int)(rlen%blen), (int)(clen%blen), sparse, (int)estNNZ );
		}
		
		//space allocation
		for( MatrixBlock b : blocks )
			if( b != null )
				if( !sparse )
					b.allocateDenseBlockUnsafe(b.getNumRows(), b.getNumColumns());
		//NOTE: no preallocation for sparse (preallocate sparserows with estnnz) in order to reduce memory footprint
		
		return blocks;
	}

	public static MatrixBlock getMatrixBlockForReuse( MatrixBlock[] blocks, int rows, int cols, int blen ) {
		int index = -1;
		if( rows==blen && cols==blen )
			index = 0;
		else if( rows==blen && cols<blen )
			index = 1;
		else if( rows<blen && cols==blen )
			index = 2;
		else //if( rows<blen && cols<blen )
			index = 3;
		return blocks[ index ];
	}
}
