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

package org.apache.sysds.runtime.functionobjects;

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.matrix.data.CTableMap;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.matrix.data.Pair;
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.apache.sysds.runtime.util.LongLongDoubleHashMap;
import org.apache.sysds.runtime.util.UtilFunctions;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

public class CTable extends ValueFunction 
{
	private static final long serialVersionUID = -5374880447194177236L;

	private static CTable singleObj = null;
	
	private CTable() {
		// nothing to do here
	}
	
	public static CTable getCTableFnObject() {
		if ( singleObj == null )
			singleObj = new CTable();
		return singleObj;
	}

	public void execute(double v1, double v2, double w, boolean ignoreZeros, CTableMap resultMap, MatrixBlock resultBlock) {
		if( resultBlock != null )
			execute(v1, v2, w, ignoreZeros, resultBlock);
		else
			execute(v1, v2, w, ignoreZeros, resultMap);
	}
	
	public void execute(double v1, double v2, double w, boolean ignoreZeros, CTableMap resultMap) {
		// If any of the values are NaN (i.e., missing) then 
		// we skip this tuple, proceed to the next tuple
		if ( Double.isNaN(v1) || Double.isNaN(v2) || Double.isNaN(w) ) {
			return;
		}
		
		// safe casts to long for consistent behavior with indexing
		long row = UtilFunctions.toLong( v1 );
		long col = UtilFunctions.toLong( v2 );
		
		// skip this entry as it does not fall within specified output dimensions
		if( ignoreZeros && row == 0 && col == 0 ) {
			return;
		}
		
		//check for incorrect ctable inputs
		if( row <= 0 || col <= 0 ) {
			throw new DMLRuntimeException("Erroneous input while computing the contingency table (one of the value <= zero): "+v1+" "+v2);
		} 
	
		//hash group-by for core ctable computation
		resultMap.aggregate(row, col, w);	
	}	

	public void execute(double v1, double v2, double w, boolean ignoreZeros, MatrixBlock ctableResult) 
	{
		// If any of the values are NaN (i.e., missing) then 
		// we skip this tuple, proceed to the next tuple
		if ( Double.isNaN(v1) || Double.isNaN(v2) || Double.isNaN(w) ) {
			return;
		}
		
		// safe casts to long for consistent behavior with indexing
		long row = UtilFunctions.toLong( v1 );
		long col = UtilFunctions.toLong( v2 );
		
		// skip this entry as it does not fall within specified output dimensions
		if( ignoreZeros && row == 0 && col == 0 ) {
			return;
		}
		
		//check for incorrect ctable inputs
		if( row <= 0 || col <= 0 ) {
			throw new DMLRuntimeException("Erroneous input while computing the contingency table (one of the value <= zero): "+v1+" "+v2);
		}
		
		// skip this entry as it does not fall within specified output dimensions
		if( row > ctableResult.getNumRows() || col > ctableResult.getNumColumns() ) {
			return;
		}
		
		//add value
		ctableResult.quickSetValue((int)row-1, (int)col-1,
				ctableResult.quickGetValue((int)row-1, (int)col-1) + w);
	}

	public int execute(int row, double v2, double w, int maxCol, int[] retIx, double[] retVals) 
	{
		// If any of the values are NaN (i.e., missing) then 
		// we skip this tuple, proceed to the next tuple
		if ( Double.isNaN(v2) || Double.isNaN(w) ) {
			return maxCol;
		}
		
		// safe casts to long for consistent behavior with indexing
		int col = UtilFunctions.toInt( v2 );
		if( col <= 0 ) {
			throw new DMLRuntimeException("Erroneous input while computing the contingency table (value <= zero): "+v2);
		} 
		
		//set weight as value (expand is guaranteed to address different cells)
		retIx[row - 1] = col - 1;
		retVals[row - 1] = w;
		
		//maintain max seen col 
		return Math.max(maxCol, col);
	}

	public Pair<MatrixIndexes,Double> execute( long row, double v2, double w ) 
	{
		// If any of the values are NaN (i.e., missing) then 
		// we skip this tuple, proceed to the next tuple
		if ( Double.isNaN(v2) || Double.isNaN(w) )
			return new Pair<>(new MatrixIndexes(-1,-1), w);
		// safe casts to long for consistent behavior with indexing
		long col = UtilFunctions.toLong( v2 );
		if( col <= 0 )
			throw new DMLRuntimeException("Erroneous input while computing the contingency table (value <= zero): "+v2);
		return new Pair<>(new MatrixIndexes(row, col), w);
	}

	/* Multithreaded CTable (F = ctable(A,B,W))
	 * Divide the input vectors into equal-sized blocks and assign each block to a task.
	 * All tasks concurrently build their own CTableMaps.
	 * Cascade merge the partial maps.
	 * TODO: Support other cases
	 */
	public void execute(MatrixBlock in1, MatrixBlock in2, MatrixBlock w, CTableMap resultMap, int k) {
		ExecutorService pool = CommonThreadPool.get(k);
		ArrayList<CTableMap> partialMaps = new ArrayList<>();
		try {
			// Assign an equal-sized blocks to each task
			List<Callable<Object>> tasks = new ArrayList<>();
			int[] blockSizes = UtilFunctions.getBlockSizes(in1.getNumRows(), k);
			// Each task builds a separate CTableMap in a lock-free manner
			for(int startRow = 0, i = 0; i < blockSizes.length; startRow += blockSizes[i], i++)
				tasks.add(getPartialCTableTask(in1, in2, w, startRow, blockSizes[i], partialMaps));
			List<Future<Object>> taskret = pool.invokeAll(tasks);
			for(var task : taskret)
				task.get();
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}

		ArrayList<CTableMap> newPartialMaps = new ArrayList<>();
		// Cascade-merge all the partial CTableMaps
		while(partialMaps.size() > 1) {
			newPartialMaps.clear();
			List<Callable<Object>> tasks = new ArrayList<>();
			int count;
			// Each task merges 2 maps and returns the merged map
			for (count=0; count+1<partialMaps.size(); count=count+2)
				tasks.add(getMergePartialCTMapsTask(partialMaps.get(count),
					partialMaps.get(count+1), newPartialMaps));

			try {
				List<Future<Object>> taskret = pool.invokeAll(tasks);
				for(var task : taskret)
					task.get();
			}
			catch(Exception ex) {
				throw new DMLRuntimeException(ex);
			}
			// Copy the remaining maps to be merged in the future iterations
			if (count < partialMaps.size())
				newPartialMaps.add(partialMaps.get(count));
			partialMaps.clear();
			partialMaps.addAll(newPartialMaps);
		}
		pool.shutdown();
		// Deep copy the last merged map into the result map
		var map = partialMaps.get(0);
		Iterator<LongLongDoubleHashMap.ADoubleEntry> iter = map.getIterator();
		while(iter.hasNext()) {
			LongLongDoubleHashMap.ADoubleEntry e = iter.next();
			resultMap.aggregate(e.getKey1(), e.getKey2(), e.value);
		}
	}

	public Callable<Object> getPartialCTableTask(MatrixBlock in1, MatrixBlock in2, MatrixBlock w,
		int startInd, int blockSize, ArrayList<CTableMap> pmaps) {
		return new PartialCTableTask(in1, in2, w, startInd, blockSize, pmaps);
	}

	public Callable<Object> getMergePartialCTMapsTask(CTableMap map1, CTableMap map2, ArrayList<CTableMap> pmaps) {
		return new MergePartialCTMaps(map1, map2, pmaps);
	}

	private static class PartialCTableTask implements Callable<Object> {
		private final MatrixBlock _in1;
		private final MatrixBlock _in2;
		private final MatrixBlock _w;
		private final int _startInd;
		private final int _blockSize;
		private final ArrayList<CTableMap> _partialCTmaps;

		protected PartialCTableTask(MatrixBlock in1, MatrixBlock in2, MatrixBlock w,
			int startRow, int blockSize, ArrayList<CTableMap> pmaps) {
			_in1 = in1;
			_in2 = in2;
			_w = w;
			_startInd = startRow;
			_blockSize = blockSize;
			_partialCTmaps = pmaps;
		}

		@Override public Object call() throws Exception {
			CTable ctable = CTable.getCTableFnObject();
			CTableMap ctmap = new CTableMap(LongLongDoubleHashMap.EntryType.INT);
			int endInd = UtilFunctions.getEndIndex(_in1.getNumRows(), _startInd, _blockSize);
			for( int i=_startInd; i<endInd; i++ )
			{
				double v1 = _in1.quickGetValue(i, 0);
				double v2 = _in2.quickGetValue(i, 0);
				double w = _w.quickGetValue(i, 0);
				ctable.execute(v1, v2, w, false, ctmap);
			}
			synchronized(_partialCTmaps) {
				_partialCTmaps.add(ctmap);
			}
			return null;
		}
	}

	private static class MergePartialCTMaps implements Callable<Object> {
		private final CTableMap _map1;
		private final CTableMap _map2;
		private final ArrayList<CTableMap> _partialCTmaps;

		protected MergePartialCTMaps(CTableMap map1, CTableMap map2, ArrayList<CTableMap> pmaps) {
			_map1 = map1;
			_map2 = map2;
			_partialCTmaps = pmaps;
		}

		private void mergeToFinal(CTableMap map, CTableMap finalMap) {
			Iterator<LongLongDoubleHashMap.ADoubleEntry> iter = map.getIterator();
			while(iter.hasNext()) {
				LongLongDoubleHashMap.ADoubleEntry e = iter.next();
				finalMap.aggregate(e.getKey1(), e.getKey2(), e.value);
			}
		}

		@Override public Object call() throws Exception {
			CTableMap mergedMap = new CTableMap(LongLongDoubleHashMap.EntryType.INT);
			mergeToFinal(_map1, mergedMap);
			mergeToFinal(_map2, mergedMap);
			synchronized(_partialCTmaps) {
				_partialCTmaps.add(mergedMap);
				return null;
			}
		}
	}
}
