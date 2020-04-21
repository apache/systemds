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

package org.tugraz.sysds.runtime.instructions.spark.functions;

import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.tugraz.sysds.runtime.DMLRuntimeException;
import org.tugraz.sysds.runtime.matrix.data.MatrixBlock;
import org.tugraz.sysds.runtime.matrix.data.MatrixIndexes;
import org.tugraz.sysds.runtime.meta.DataCharacteristics;
import org.tugraz.sysds.runtime.util.UtilFunctions;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.Iterator;

public class ExtractBlockForBinaryReblock implements PairFlatMapFunction<Tuple2<MatrixIndexes,MatrixBlock>, MatrixIndexes, MatrixBlock> 
{
	private static final long serialVersionUID = -762987655085029215L;
	
	private final long rlen, clen; 
	private final int in_blen;
	private final int out_blen;
	
	public ExtractBlockForBinaryReblock(DataCharacteristics mcIn, DataCharacteristics mcOut) {
		rlen = mcIn.getRows();
		clen = mcIn.getCols();
		in_blen = mcIn.getBlocksize();
		out_blen = mcOut.getBlocksize();
		
		//sanity check block sizes
		if(in_blen <= 0 || out_blen <= 0)
			throw new DMLRuntimeException("Block sizes not unknown:" + in_blen + "," +  out_blen);
	}
	
	@Override
	public Iterator<Tuple2<MatrixIndexes, MatrixBlock>> call(Tuple2<MatrixIndexes, MatrixBlock> arg0) 
		throws Exception 
	{
		MatrixIndexes ixIn = arg0._1();
		MatrixBlock in = arg0._2();
		
		final long startRowGlobalCellIndex = UtilFunctions.computeCellIndex(ixIn.getRowIndex(), in_blen, 0);
		final long endRowGlobalCellIndex = getEndGlobalIndex(ixIn.getRowIndex(), true, true);
		final long startColGlobalCellIndex = UtilFunctions.computeCellIndex(ixIn.getColumnIndex(), in_blen, 0);
		final long endColGlobalCellIndex = getEndGlobalIndex(ixIn.getColumnIndex(), true, false);
		
		final long out_startRowBlockIndex = UtilFunctions.computeBlockIndex(startRowGlobalCellIndex, out_blen);
		final long out_endRowBlockIndex = UtilFunctions.computeBlockIndex(endRowGlobalCellIndex, out_blen);
		final long out_startColBlockIndex = UtilFunctions.computeBlockIndex(startColGlobalCellIndex, out_blen);
		final long out_endColBlockIndex = UtilFunctions.computeBlockIndex(endColGlobalCellIndex, out_blen);
		final boolean aligned = out_blen%in_blen==0 && out_blen%in_blen==0; //e.g, 1K -> 2K
		
		ArrayList<Tuple2<MatrixIndexes, MatrixBlock>> retVal = new ArrayList<>();
		for(long i = out_startRowBlockIndex; i <= out_endRowBlockIndex; i++) {
			for(long j = out_startColBlockIndex; j <= out_endColBlockIndex; j++) {
				MatrixIndexes indx = new MatrixIndexes(i, j);
				final int new_lrlen = UtilFunctions.computeBlockSize(rlen, i, out_blen);
				final int new_lclen = UtilFunctions.computeBlockSize(clen, j, out_blen);
				MatrixBlock blk = new MatrixBlock(new_lrlen, new_lclen, true);
				if( in.isEmptyBlock(false) ) continue;
				
				final long rowLower = Math.max(UtilFunctions.computeCellIndex(i, out_blen, 0), startRowGlobalCellIndex);
				final long rowUpper = Math.min(getEndGlobalIndex(i, false, true), endRowGlobalCellIndex);
				final long colLower = Math.max(UtilFunctions.computeCellIndex(j, out_blen, 0), startColGlobalCellIndex);
				final long colUpper = Math.min(getEndGlobalIndex(j, false, false), endColGlobalCellIndex);
				final int aixi = UtilFunctions.computeCellInBlock(rowLower, in_blen);
				final int aixj = UtilFunctions.computeCellInBlock(colLower, in_blen);
				final int cixi = UtilFunctions.computeCellInBlock(rowLower, out_blen);
				final int cixj = UtilFunctions.computeCellInBlock(colLower, out_blen);
				
				if( aligned ) {
					blk.appendToSparse(in, cixi, cixj);
					blk.setNonZeros(in.getNonZeros());
				}
				else { //general case
					for(int i2 = 0; i2 <= (int)(rowUpper-rowLower); i2++)
						for(int j2 = 0; j2 <= (int)(colUpper-colLower); j2++)
							blk.appendValue(cixi+i2, cixj+j2, in.quickGetValue(aixi+i2, aixj+j2));
				}
				retVal.add(new Tuple2<>(indx, blk));
			}
		}
		
		return retVal.iterator();
	}

	private long getEndGlobalIndex(long blockIndex, boolean isIn, boolean isRow) {
		//determine dimension and block sizes
		long len = isRow ? rlen : clen;
		int blen = isIn ? (isRow ? in_blen : in_blen) : (isRow ? out_blen : out_blen);
		
		//compute 1-based global cell index in block
		int new_len = UtilFunctions.computeBlockSize(len, blockIndex, blen);
		return UtilFunctions.computeCellIndex(blockIndex, blen, new_len-1);
	}
}
