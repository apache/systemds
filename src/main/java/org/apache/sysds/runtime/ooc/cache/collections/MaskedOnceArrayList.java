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

package org.apache.sysds.runtime.ooc.cache.collections;

import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.util.function.Consumer;

public class MaskedOnceArrayList<T> {
	private static final VarHandle PARTITIONS;
	private static final VarHandle PARTITION = MethodHandles.arrayElementVarHandle(MaskedOnceArray[].class);
	private static final int DEFAULT_PARTITION_SIZE = 1024;

	static {
		try {
			PARTITIONS = MethodHandles.lookup().findVarHandle(MaskedOnceArrayList.class, "_partitions",
				MaskedOnceArray[].class);
		}
		catch(ReflectiveOperationException e) {
			throw new ExceptionInInitializerError(e);
		}
	}

	private final int _partitionSize;
	private final int _partitionBits;
	private final int _partitionMask;

	@SuppressWarnings("rawtypes")
	private volatile MaskedOnceArray[] _partitions;

	public MaskedOnceArrayList() {
		this(DEFAULT_PARTITION_SIZE);
	}

	public MaskedOnceArrayList(int partitionSize) {
		validatePartitionSize(partitionSize);
		_partitionSize = partitionSize;
		_partitionBits = Integer.numberOfTrailingZeros(partitionSize);
		_partitionMask = partitionSize - 1;
		_partitions = new MaskedOnceArray[1];
	}

	@SuppressWarnings("rawtypes")
	public boolean put(int i, T value) {
		checkIndex(i);
		if(value == null)
			return clear(i);
		int partitionIndex = partitionIndex(i);
		int offset = offsetInPartition(i);
		while(true) {
			MaskedOnceArray[] partitions = ensurePartitionCapacity(partitionIndex);
			MaskedOnceArray<T> partition = partitionAt(partitions, partitionIndex);
			boolean changed = partition.put(offset, value);
			if(PARTITION.getAcquire(partitions, partitionIndex) == partition && !partition.isRetired())
				return changed;
		}
	}

	@SuppressWarnings("rawtypes")
	public boolean clear(int i) {
		checkIndex(i);
		int partition = partitionIndex(i);
		MaskedOnceArray[] partitions = (MaskedOnceArray[]) PARTITIONS.getAcquire(this);
		if(partition < partitions.length)
			return clear(partitions, partition, offsetInPartition(i));
		return false;
	}

	@SuppressWarnings({"unchecked", "rawtypes"})
	public T get(int i) {
		checkIndex(i);
		int partition = partitionIndex(i);
		MaskedOnceArray[] partitions = (MaskedOnceArray[]) PARTITIONS.getAcquire(this);
		if(partition >= partitions.length)
			return null;
		MaskedOnceArray p = (MaskedOnceArray) PARTITION.getAcquire(partitions, partition);
		return p == null ? null : (T) p.get(offsetInPartition(i));
	}

	@SuppressWarnings("rawtypes")
	public void setLive(int i) {
		checkIndex(i);
		int partitionIndex = partitionIndex(i);
		MaskedOnceArray[] partitions = ensurePartitionCapacity(partitionIndex);
		partitionAt(partitions, partitionIndex).setLive(offsetInPartition(i));
	}

	@SuppressWarnings("rawtypes")
	public void clearLive(int i) {
		checkIndex(i);
		int partition = partitionIndex(i);
		MaskedOnceArray[] partitions = (MaskedOnceArray[]) PARTITIONS.getAcquire(this);
		if(partition < partitions.length) {
			MaskedOnceArray p = (MaskedOnceArray) PARTITION.getAcquire(partitions, partition);
			if(p != null)
				p.clearLive(offsetInPartition(i));
		}
	}

	@SuppressWarnings("rawtypes")
	public int capacity() {
		MaskedOnceArray[] partitions = (MaskedOnceArray[]) PARTITIONS.getAcquire(this);
		return partitions.length * _partitionSize;
	}

	@SuppressWarnings({"rawtypes", "unchecked"})
	public void forEachLive(IndexedObjectPredicate<? super T> action, boolean reversed) {
		MaskedOnceArray[] partitions = (MaskedOnceArray[]) PARTITIONS.getAcquire(this);
		if(reversed) {
			for(int i = partitions.length - 1; i >= 0; i--) {
				MaskedOnceArray partition = (MaskedOnceArray) PARTITION.getAcquire(partitions, i);
				if(partition != null)
					partition.forEachLive(action, true, i * _partitionSize);
			}
		}
		else {
			for(int i = 0; i < partitions.length; i++) {
				MaskedOnceArray partition = (MaskedOnceArray) PARTITION.getAcquire(partitions, i);
				if(partition != null)
					partition.forEachLive(action, false, i * _partitionSize);
			}
		}
	}

	@SuppressWarnings({"rawtypes", "unchecked"})
	public void forEachVisible(Consumer<? super T> action) {
		MaskedOnceArray[] partitions = (MaskedOnceArray[]) PARTITIONS.getAcquire(this);
		for(int i = 0; i < partitions.length; i++) {
			MaskedOnceArray partition = (MaskedOnceArray) PARTITION.getAcquire(partitions, i);
			if(partition != null)
				partition.forEachVisible(action);
		}
	}

	@SuppressWarnings("rawtypes")
	private MaskedOnceArray[] ensurePartitionCapacity(int partitionIndex) {
		MaskedOnceArray[] partitions = (MaskedOnceArray[]) PARTITIONS.getAcquire(this);
		while(partitionIndex >= partitions.length) {
			MaskedOnceArray[] bigger = growPartitions(partitions, partitionIndex + 1);
			if(PARTITIONS.compareAndSet(this, partitions, bigger))
				partitions = bigger;
			else
				partitions = (MaskedOnceArray[]) PARTITIONS.getAcquire(this);
		}
		return partitions;
	}

	@SuppressWarnings({"unchecked", "rawtypes"})
	private MaskedOnceArray<T> partitionAt(MaskedOnceArray[] partitions, int partitionIndex) {
		MaskedOnceArray<T> partition;
		while((partition = (MaskedOnceArray<T>) PARTITION.getAcquire(partitions, partitionIndex)) == null ||
			partition.isRetired()) {
			if(partition != null) {
				PARTITION.compareAndSet(partitions, partitionIndex, partition, null);
				continue;
			}
			MaskedOnceArray<T> newPartition = new MaskedOnceArray<>(_partitionSize);
			if(PARTITION.compareAndSet(partitions, partitionIndex, null, newPartition))
				return newPartition;
		}
		return partition;
	}

	@SuppressWarnings("rawtypes")
	private boolean clear(MaskedOnceArray[] partitions, int partitionIndex, int offset) {
		MaskedOnceArray partition = (MaskedOnceArray) PARTITION.getAcquire(partitions, partitionIndex);
		if(partition == null)
			return false;
		boolean changed = partition.clear(offset);
		if(partition.tryRetireIfEmpty())
			PARTITION.compareAndSet(partitions, partitionIndex, partition, null);
		return changed;
	}

	@SuppressWarnings("rawtypes")
	private MaskedOnceArray[] growPartitions(MaskedOnceArray[] partitions, int minLength) {
		int newLength = partitions.length;
		while(newLength < minLength) {
			if(newLength > Integer.MAX_VALUE / 2)
				throw new IllegalStateException("MaskedOnceArrayList capacity overflow");
			newLength <<= 1;
		}

		MaskedOnceArray[] bigger = new MaskedOnceArray[newLength];
		System.arraycopy(partitions, 0, bigger, 0, partitions.length);
		return bigger;
	}

	private int partitionIndex(int index) {
		return index >>> _partitionBits;
	}

	private int offsetInPartition(int index) {
		return index & _partitionMask;
	}

	private static void validatePartitionSize(int partitionSize) {
		if(partitionSize < 64 || (partitionSize & (partitionSize - 1)) != 0) {
			throw new IllegalArgumentException(
				"partitionSize must be a power of two and at least 64: " + partitionSize);
		}
	}

	private static void checkIndex(int i) {
		if(i < 0)
			throw new IndexOutOfBoundsException("Negative index: " + i);
	}
}
