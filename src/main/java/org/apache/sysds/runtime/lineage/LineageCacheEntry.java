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

import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
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
	
	public LineageCacheEntry(LineageItem key, DataType dt, MatrixBlock Mval, ScalarObject Sval, long computetime) {
		_key = key;
		_dt = dt;
		_MBval = Mval;
		_SOval = Sval;
		_computeTime = computetime;
		_status = isNullVal() ? LineageCacheStatus.EMPTY : LineageCacheStatus.CACHED;
		_nextEntry = null;
		_origItem = null;
	}
	
	protected synchronized void setCacheStatus(LineageCacheStatus st) {
		_status = st;
	}

	public synchronized MatrixBlock getMBValue() {
		try {
			//wait until other thread completes operation
			//in order to avoid redundant computation
			while( _MBval == null ) {
				wait();
			}
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
			while( _SOval == null ) {
				wait();
			}
			return _SOval;
		}
		catch( InterruptedException ex ) {
			throw new DMLRuntimeException(ex);
		}
	}
	
	public synchronized LineageCacheStatus getCacheStatus() {
		return _status;
	}
	
	public synchronized long getSize() {
		return ((_MBval != null ? _MBval.getInMemorySize() : 0) + (_SOval != null ? _SOval.getSize() : 0));
	}
	
	public boolean isNullVal() {
		return(_MBval == null && _SOval == null);
	}
	
	public boolean isMatrixValue() {
		return _dt.isMatrix();
	}
	
	public synchronized void setValue(MatrixBlock val, long computetime) {
		_MBval = val;
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
		_computeTime = computetime;
		_status = isNullVal() ? LineageCacheStatus.EMPTY : LineageCacheStatus.CACHED;
		//resume all threads waiting for val
		notifyAll();
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
		_timestamp =  System.currentTimeMillis() - LineageCacheEviction.getStartTimestamp();
		if (_timestamp < 0)
			throw new DMLRuntimeException ("Execution timestamp shouldn't be -ve. Key: "+_key);
		recomputeScore();
	}
	
	protected synchronized long getTimestamp() {
		return _timestamp;
	}
	
	private void recomputeScore() {
		// Gather the weights for scoring components
		double w1 = LineageCacheConfig.WEIGHTS[0];
		double w2 = LineageCacheConfig.WEIGHTS[1];
		// Generate scores
		score = w1*(((double)_computeTime)/getSize()) + w2*getTimestamp();
	}
}
