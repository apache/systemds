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

import jcuda.Pointer;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.instructions.spark.data.RDDObject;
import org.apache.sysds.runtime.lineage.LineageCacheConfig.LineageCacheStatus;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.meta.MetaData;

public class LineageCacheEntry {
	protected final LineageItem _key;
	protected final DataType _dt;
	protected MatrixBlock _MBval;
	protected ScalarObject _SOval;
	protected byte[] _serialBytes; // serialized bytes of a federated response
	protected long _computeTime;
	protected long _timestamp = 0;
	protected LineageCacheStatus _status;
	protected LineageCacheEntry _nextEntry;
	protected LineageItem _origItem;
	private String _outfile = null;
	protected double score;
	protected GPUPointer _gpuPointer;

	protected RDDObject _rddObject;
	
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
		_gpuPointer = null;
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

	public synchronized RDDObject getRDDObject() {
		try {
			//wait until other thread completes operation
			//in order to avoid redundant computation
			while(_status == LineageCacheStatus.EMPTY) {
				wait();
			}
			//comes here if data is placed or the entry is removed by the running thread
			return _rddObject;
		}
		catch( InterruptedException ex ) {
			throw new DMLRuntimeException(ex);
		}
	}

	public synchronized byte[] getSerializedBytes() {
		try {
			// wait until other thread completes operation
			// in order to avoid redundant computation
			while(_status == LineageCacheStatus.EMPTY) {
				wait();
			}
			// comes here if data is placed or the entry is removed by the running thread
			return _serialBytes;
		}
		catch( InterruptedException ex ) {
			throw new DMLRuntimeException(ex);
		}
	}

	public synchronized Pointer getGPUPointer() {
		try {
			//wait until other thread completes operation
			//in order to avoid redundant computation
			while(_status == LineageCacheStatus.EMPTY) {
				wait();
			}
			//comes here if data is placed or the entry is removed by the running thread
			return _gpuPointer.getPointer();
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
		if (_gpuPointer != null)
			size += _gpuPointer.getPointerSize();
		if (_rddObject != null)
			//Return total cached size in the executors
			size += SparkExecutionContext.getMemCachedRDDSize(_rddObject.getRDD().id());
		return size;
	}
	
	public boolean isNullVal() {
		return(_MBval == null && _SOval == null && _gpuPointer == null && _serialBytes == null && _rddObject == null);
	}
	
	public boolean isMatrixValue() {
		return _dt.isMatrix() && _rddObject == null && _gpuPointer == null;
	}

	public boolean isScalarValue() {
		return _dt.isScalar() && _rddObject == null && _gpuPointer == null;
	}

	public boolean isLocalObject() {
		return isMatrixValue() || isScalarValue();
	}

	public boolean isRDDPersist() {
		return _rddObject != null;
	}

	public boolean isGPUObject() {
		return _gpuPointer!= null;
	}

	public synchronized boolean isDensePointer() {
		if (!isGPUObject())
			return false;
		return _gpuPointer.isDensepointer();
	}

	public boolean isSerializedBytes() {
		return _dt.isUnknown() && _key.getOpcode().equals(LineageItemUtils.SERIALIZATION_OPCODE);
	}

	public synchronized void setValue(MatrixBlock val, long computetime) {
		_MBval = val;
		_gpuPointer = null;  //Matrix block and gpu object cannot coexist
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
		_gpuPointer = null;  //scalar and gpu object cannot coexist
		_computeTime = computetime;
		_status = isNullVal() ? LineageCacheStatus.EMPTY : LineageCacheStatus.CACHED;
		//resume all threads waiting for val
		notifyAll();
	}
	
	public synchronized void setGPUValue(Pointer ptr, long size, MetaData md, long computetime) {
		_gpuPointer = new GPUPointer(ptr, size, md);
		_computeTime = computetime;
		_status = isNullVal() ? LineageCacheStatus.EMPTY : LineageCacheStatus.GPUCACHED;
		//resume all threads waiting for val
		notifyAll();
	}

	public synchronized void setRDDValue(RDDObject rdd, long computetime) {
		_rddObject = rdd;
		_computeTime = computetime;
		//_status = isNullVal() ? LineageCacheStatus.EMPTY : LineageCacheStatus.CACHED;
		_status = isNullVal() ? LineageCacheStatus.EMPTY : LineageCacheStatus.TOPERSISTRDD;
		//resume all threads waiting for val
		notifyAll();
	}

	public synchronized void setValue(byte[] serialBytes, long computetime) {
		_serialBytes = serialBytes;
		_computeTime = computetime;
		_status = isNullVal() ? LineageCacheStatus.EMPTY : LineageCacheStatus.CACHED;
		// resume all threads waiting for val
		notifyAll();
	}

	public synchronized void copyValueFrom(LineageCacheEntry src, long computetime) {
		_MBval = src._MBval;
		_SOval = src._SOval;
		_gpuPointer = src._gpuPointer;
		_rddObject = src._rddObject;
		_computeTime = src._computeTime;
		_status = isNullVal() ? LineageCacheStatus.EMPTY : LineageCacheStatus.CACHED;
		// resume all threads waiting for val
		notifyAll();
	}

	public synchronized DataCharacteristics getDataCharacteristics() {
		return _gpuPointer.getDataCharacteristics();
	}

	protected synchronized void setNullValues() {
		_MBval = null;
		_SOval = null;
		_serialBytes = null;
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
			double w1 = LineageCacheConfig.WEIGHTS[0];
			int missCount = 1 + removeList.get(_key);
			long size = getSize();
			if (isLocalObject())
				score = score + (w1*(((double)_computeTime)/getSize()) * missCount);
		}
	}

	protected synchronized void initiateScoreSpark(Map<LineageItem, Integer> removeList, long estimatedSize) {
		// Set timestamp
		_timestamp =  System.currentTimeMillis() - LineageCacheEviction.getStartTimestamp();
		if (_timestamp < 0)
			throw new DMLRuntimeException ("Execution timestamp shouldn't be -ve. Key: "+_key);

		// Gather the weights for scoring components
		double w1 = LineageCacheConfig.WEIGHTS[0];
		double w2 = LineageCacheConfig.WEIGHTS[1];
		double w3 = LineageCacheConfig.WEIGHTS[2];
		// Generate initial score
		int computeGroup = LineageCacheConfig.getComputeGroup(_key.getOpcode());
		int refCount = Math.max(_rddObject.getMaxReferenceCount(), 1);
		score = w1*(((double)computeGroup*refCount)/estimatedSize) + w2*getTimestamp() + w3*(((double)1)/getDagHeight());
	}
	
	protected synchronized void updateScore(boolean add) {
		// Update score to emulate computeTime scaling by cache hit
		double w1 = LineageCacheConfig.WEIGHTS[0];
		long size = getSize();
		int sign = add ? 1: -1;
		 if(isLocalObject())
			 score = score + sign * w1 * (((double) _computeTime) / size);
		 if(isRDDPersist() && size != 0) {  //size == 0 means not persisted yet
			 int computeGroup = LineageCacheConfig.getComputeGroup(_key.getOpcode());
			 score = score + sign * w1 * (((double) computeGroup) / size);
		 }
	}
	
	protected synchronized long getTimestamp() {
		return _timestamp;
	}
	
	protected synchronized long getDagHeight() {
		return _key.getHeight();
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
		long size = getSize();
		if (isLocalObject())
			score = w1*(((double)_computeTime)/size) + w2*getTimestamp() + w3*(((double)1)/getDagHeight());
		if (isRDDPersist() && size != 0) {  //size == 0 means not persisted yet
			int computeGroup = LineageCacheConfig.getComputeGroup(_key.getOpcode());
			int refCount = Math.max(_rddObject.getMaxReferenceCount(), 1);
			score = w1*(((double)computeGroup*refCount)/size) + w2*getTimestamp() + w3*(((double)1)/getDagHeight());
		}
	}

	static class GPUPointer {
		private Pointer _pointer;
		private long _allocatedSize; //bytes
		private MetaData _metadata;

		public GPUPointer(Pointer pointer, long size, MetaData metadata) {
			_pointer = pointer;
			_allocatedSize = size;
			_metadata = metadata;
		}

		protected long getPointerSize() {
			return _allocatedSize;
		}

		protected Pointer getPointer() {
			return _pointer;
		}

		protected DataCharacteristics getDataCharacteristics() {
			return _metadata.getDataCharacteristics();
		}

		protected boolean isDensepointer() {
			return true;
			// TODO: Support sparse pointer caching
		}
	}
}
