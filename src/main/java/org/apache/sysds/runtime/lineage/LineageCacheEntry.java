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

package org.apache.sysds.runtime.lineage;

import java.util.Map;

import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.instructions.gpu.context.GPUObject;
import org.apache.sysds.runtime.lineage.LineageCacheConfig.LineageCacheStatus;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public class LineageCacheEntry {
	protected final LineageItem _key;
	protected final DataType _dt;
	protected MatrixBlock _MBval;
	protected ScalarObject _SOval;
	protected long _computeTime;
	protected long _timestamp = 0;
	protected LineageCacheStatus _status;
	protected LineageCacheEntry _nextEntry;
	protected LineageItem _origItem;
	private String _outfile = null;
	protected double score;
	protected GPUObject _gpuObject;
	
	public LineageCacheEntry(LineageItem key, DataType dt, MatrixBlock Mval, ScalarObject Sval, long computetime) {
		_key = key;
		_dt = dt;
		_MBval = Mval;
		_SOval = Sval;
		_computeTime = computetime;
		_status = isNullVal() ? LineageCacheStatus.EMPTY : LineageCacheStatus.CACHED;
		_nextEntry = null;
		_origItem = null;
		_outfile = null;
		_gpuObject = null;
	}
	
	protected synchronized void setCacheStatus(LineageCacheStatus st) {
		_status = st;
	}

	public synchronized MatrixBlock getMBValue() {
		try {
			//wait until other thread completes operation
			//in order to avoid redundant computation
			while(_status == LineageCacheStatus.EMPTY) {
				wait();
			}
			//comes here if data is placed or the entry is removed by the running thread
			return _MBval;
		}
		catch( InterruptedException ex ) {
			throw new DMLRuntimeException(ex);
		}
	}

	public synchronized ScalarObject getSOValue() {
		try {
			//wait until other thread completes operation
			//in order to avoid redundant computation
			while(_status == LineageCacheStatus.EMPTY) {
				wait();
			}
			//comes here if data is placed or the entry is removed by the running thread
			return _SOval;
		}
		catch( InterruptedException ex ) {
			throw new DMLRuntimeException(ex);
		}
	}
	
	public synchronized LineageCacheStatus getCacheStatus() {
		return _status;
	}
	
	protected synchronized void removeAndNotify() {
		//Set the status to NOTCACHED (not cached anymore) and wake up the sleeping threads
		if (_status != LineageCacheStatus.EMPTY)
			return;
		_status = LineageCacheStatus.NOTCACHED;
		notifyAll();
	}
	
	public synchronized long getSize() {
		long size = 0;
		if (_MBval != null)
			size += _MBval.getInMemorySize();
		if (_SOval != null)
			size += _SOval.getSize();
		if (_gpuObject != null)
			size += _gpuObject.getSizeOnDevice();
		return size;
	}
	
	public boolean isNullVal() {
		return(_MBval == null && _SOval == null && _gpuObject == null);
	}
	
	public boolean isMatrixValue() {
		return _dt.isMatrix();
	}

	public boolean isScalarValue() {
		return _dt.isScalar();
	}
	
	public synchronized void setValue(MatrixBlock val, long computetime) {
		_MBval = val;
		_gpuObject = null;  //Matrix block and gpu object cannot coexist
		_computeTime = computetime;
		_status = isNullVal() ? LineageCacheStatus.EMPTY : LineageCacheStatus.CACHED;
		//resume all threads waiting for val
		notifyAll();
	}
	
	public synchronized void setValue(MatrixBlock val) {
		setValue(val, _computeTime);
	}

	public synchronized void setValue(ScalarObject val, long computetime) {
		_SOval = val;
		_gpuObject = null;  //scalar and gpu object cannot coexist
		_computeTime = computetime;
		_status = isNullVal() ? LineageCacheStatus.EMPTY : LineageCacheStatus.CACHED;
		//resume all threads waiting for val
		notifyAll();
	}
	
	public synchronized void setGPUValue(GPUObject gpuObj, long computetime) {
		gpuObj.setIsLinCached(true);
		_gpuObject = gpuObj;
		_computeTime = computetime;
		_status = isNullVal() ? LineageCacheStatus.EMPTY : LineageCacheStatus.GPUCACHED;
		//resume all threads waiting for val
		notifyAll();
	}
	
	public synchronized GPUObject getGPUObject() {
		return _gpuObject;
	}
	
	protected synchronized void setNullValues() {
		_MBval = null;
		_SOval = null;
		_status = LineageCacheStatus.EMPTY;
	}
	
	protected synchronized void setOutfile(String outfile) {
		_outfile = outfile;
	}
	
	protected synchronized String getOutfile() {
		return _outfile;
	}
	
	protected synchronized void setTimestamp() {
		if (_timestamp != 0)
			return;
		
		_timestamp =  System.currentTimeMillis() - LineageCacheEviction.getStartTimestamp();
		if (_timestamp < 0)
			throw new DMLRuntimeException ("Execution timestamp shouldn't be -ve. Key: "+_key);
		recomputeScore();
	}

	protected synchronized void updateTimestamp() {
		_timestamp =  System.currentTimeMillis() - LineageCacheEviction.getStartTimestamp();
		if (_timestamp < 0)
			throw new DMLRuntimeException ("Execution timestamp shouldn't be -ve. Key: "+_key);
		recomputeScore();
	}
	
	protected synchronized void computeScore(Map<LineageItem, Integer> removeList) {
		// Set timestamp and compute initial score
		setTimestamp();

		// Update score to emulate computeTime scaling by #misses
		if (removeList.containsKey(_key) && LineageCacheConfig.isCostNsize()) {
			//score = score * (1 + removeList.get(_key));
			double w1 = LineageCacheConfig.WEIGHTS[0];
			int missCount = 1 + removeList.get(_key);
			score = score + (w1*(((double)_computeTime)/getSize()) * missCount);
		}
	}
	
	protected synchronized void updateScore() {
		// Update score to emulate computeTime scaling by cache hit
		//score *= 2;
		double w1 = LineageCacheConfig.WEIGHTS[0];
		score = score + w1*(((double)_computeTime)/getSize());
	}
	
	protected synchronized long getTimestamp() {
		return _timestamp;
	}
	
	protected synchronized long getDagHeight() {
		return _key.getDistLeaf2Node();
	}
	
	protected synchronized double getCostNsize() {
		return ((double)_computeTime)/getSize();
	}
	
	private void recomputeScore() {
		// Gather the weights for scoring components
		double w1 = LineageCacheConfig.WEIGHTS[0];
		double w2 = LineageCacheConfig.WEIGHTS[1];
		double w3 = LineageCacheConfig.WEIGHTS[2];
		// Generate scores
		score = w1*(((double)_computeTime)/getSize()) + w2*getTimestamp() + w3*(((double)1)/getDagHeight());
	}
}
