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


package org.apache.sysml.runtime.matrix.data;

public abstract class SparseBlockFactory
{

	public static SparseBlock createSparseBlock(int rlen) {
		return createSparseBlock(MatrixBlock.DEFAULT_SPARSEBLOCK, rlen);
	}

	public static SparseBlock createSparseBlock( SparseBlock.Type type, int rlen ) {
		switch( type ) {
			case MCSR: return new SparseBlockMCSR(rlen, -1);
			case CSR: return new SparseBlockCSR(rlen);
			case COO: return new SparseBlockCOO(rlen);
			default:
				throw new RuntimeException("Unexpected sparse block type: "+type.toString());
		}
	}

	public static SparseBlock copySparseBlock( SparseBlock.Type type, SparseBlock sblock, boolean forceCopy )
	{
		//sanity check for empty inputs
		if( sblock == null )
			return null;
		
		//check for existing target type
		if( !forceCopy && isSparseBlockType(sblock, type) ){
			return sblock;
		}
		
		//create target sparse block
		switch( type ) {
			case MCSR: return new SparseBlockMCSR(sblock);
			case CSR: return new SparseBlockCSR(sblock);
			case COO: return new SparseBlockCOO(sblock);
			default:
				throw new RuntimeException("Unexpected sparse block type: "+type.toString());
		}
	}
	
	public static boolean isSparseBlockType(SparseBlock sblock, SparseBlock.Type type) {
		return (getSparseBlockType(sblock) == type);
	}
	
	public static SparseBlock.Type getSparseBlockType(SparseBlock sblock) {
		return (sblock instanceof SparseBlockMCSR) ? SparseBlock.Type.MCSR :
			(sblock instanceof SparseBlockCSR) ? SparseBlock.Type.CSR : 
			(sblock instanceof SparseBlockCOO) ? SparseBlock.Type.COO : null;
	}

	public static long estimateSizeSparseInMemory(SparseBlock.Type type, long nrows, long ncols, double sparsity) {
		switch( type ) {
			case MCSR: return SparseBlockMCSR.estimateMemory(nrows, ncols, sparsity);
			case CSR: return SparseBlockCSR.estimateMemory(nrows, ncols, sparsity);
			case COO: return SparseBlockCOO.estimateMemory(nrows, ncols, sparsity);
			default:
				throw new RuntimeException("Unexpected sparse block type: "+type.toString());
		}
	}
}
