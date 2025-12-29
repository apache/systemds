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

package org.apache.sysds.runtime.instructions.ooc;

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.parfor.LocalTaskQueue;
import org.apache.sysds.runtime.controlprogram.parfor.util.IDSequence;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import shaded.parquet.it.unimi.dsi.fastutil.ints.IntArrayList;

import java.util.HashMap;
import java.util.Map;

/**
 * A wrapper around LocalTaskQueue to consume the source stream and reset to
 * consume again for other operators.
 *
 */
public class CachingStream implements OOCStreamable<IndexedMatrixValue> {

	public static final IDSequence _streamSeq = new IDSequence();

	// original live stream
	private final OOCStream<IndexedMatrixValue> _source;
	private final IntArrayList _consumptionCounts = new IntArrayList();

	// stream identifier
	private final long _streamId;

	// block counter
	private int _numBlocks = 0;

	private Runnable[] _subscribers;

	// state flags
	private boolean _cacheInProgress = true; // caching in progress, in the first pass.
	private Map<MatrixIndexes, Integer> _index;

	private DMLRuntimeException _failure;

	private boolean deletable = false;
	private int maxConsumptionCount = 0;
	private int cachePins = 0;

	public CachingStream(OOCStream<IndexedMatrixValue> source) {
		this(source, _streamSeq.getNextID());
	}

	public CachingStream(OOCStream<IndexedMatrixValue> source, long streamId) {
		_source = source;
		_streamId = streamId;
		source.setSubscriber(tmp -> {
			try {
				final IndexedMatrixValue task = tmp.get();
				int blk;
				Runnable[] mSubscribers;

				synchronized (this) {
					if(task != LocalTaskQueue.NO_MORE_TASKS) {
						if (!_cacheInProgress)
							throw new DMLRuntimeException("Stream is closed");
						OOCEvictionManager.put(_streamId, _numBlocks, task);
						if (_index != null)
							_index.put(task.getIndexes(), _numBlocks);
						blk = _numBlocks;
						_numBlocks++;
						_consumptionCounts.add(0);
						notifyAll();
					}
					else {
						_cacheInProgress = false; // caching is complete
						notifyAll();
						blk = -1;
					}

					mSubscribers = _subscribers;
				}

				if(mSubscribers != null) {
					for(Runnable mSubscriber : mSubscribers)
						mSubscriber.run();

					if (blk == -1) {
						synchronized (this) {
							_subscribers = null;
						}
					}
				}
			} catch (DMLRuntimeException e) {
				// Propagate failure to subscribers
				_failure = e;
				synchronized (this) {
					notifyAll();
				}

				Runnable[] mSubscribers = _subscribers;
				if(mSubscribers != null) {
					for(Runnable mSubscriber : mSubscribers) {
						try {
							mSubscriber.run();
						} catch (Exception ignored) {
						}
					}
				}
			}
		});
	}

	public synchronized void scheduleDeletion() {
		deletable = true;
		if (_cacheInProgress && maxConsumptionCount == 0)
			throw new DMLRuntimeException("Cannot have a caching stream with no listeners");
		for (int i = 0; i < _consumptionCounts.size(); i++) {
			tryDeleteBlock(i);
		}
	}

	public String toString() {
		return "CachingStream@" + _streamId;
	}

	private synchronized void tryDeleteBlock(int i) {
		if (cachePins > 0)
			return; // Block deletion is prevented

		int count = _consumptionCounts.getInt(i);
		if (count > maxConsumptionCount)
			throw new DMLRuntimeException("Cannot have more than " + maxConsumptionCount + " consumptions.");
		if (count == maxConsumptionCount)
			OOCEvictionManager.forget(_streamId, i);
	}

	public synchronized IndexedMatrixValue get(int idx) throws InterruptedException {
		while (true) {
			if (_failure != null)
				throw _failure;
			else if (idx < _numBlocks) {
				IndexedMatrixValue out = OOCEvictionManager.get(_streamId, idx);

				if (_index != null) // Ensure index is up to date
					_index.putIfAbsent(out.getIndexes(), idx);

				int newCount = _consumptionCounts.getInt(idx)+1;

				if (newCount > maxConsumptionCount)
					throw new DMLRuntimeException("Consumer overflow! Expected: " + maxConsumptionCount);

				_consumptionCounts.set(idx, newCount);

				if (deletable)
					tryDeleteBlock(idx);

				return out;
			} else if (!_cacheInProgress)
				return (IndexedMatrixValue)LocalTaskQueue.NO_MORE_TASKS;

			wait();
		}
	}

	public synchronized int findCachedIndex(MatrixIndexes idx) {
		return _index.get(idx);
	}

	public synchronized IndexedMatrixValue findCached(MatrixIndexes idx) {
		int mIdx = _index.get(idx);
		int newCount = _consumptionCounts.getInt(mIdx)+1;
		if (newCount > maxConsumptionCount)
			throw new DMLRuntimeException("Consumer overflow in " + _streamId + "_" + mIdx + ". Expected: " + maxConsumptionCount);
		_consumptionCounts.set(mIdx, newCount);

		IndexedMatrixValue imv = OOCEvictionManager.get(_streamId, mIdx);

		if (deletable)
			tryDeleteBlock(mIdx);

		return imv;
	}

	/**
	 * Finds a cached item without counting it as a consumption.
	 */
	public synchronized IndexedMatrixValue peekCached(MatrixIndexes idx) {
		int mIdx = _index.get(idx);
		return OOCEvictionManager.get(_streamId, mIdx);
	}

	public synchronized void activateIndexing() {
		if (_index == null)
			_index = new HashMap<>();
	}

	@Override
	public OOCStream<IndexedMatrixValue> getReadStream() {
		return new PlaybackStream(this);
	}

	@Override
	public OOCStream<IndexedMatrixValue> getWriteStream() {
		return _source.getWriteStream();
	}

	@Override
	public boolean isProcessed() {
		return false;
	}

	public void setSubscriber(Runnable subscriber, boolean incrConsumers) {
		if (deletable)
			throw new DMLRuntimeException("Cannot register a new subscriber on " + this + " because has been flagged for deletion");

		int mNumBlocks;
		boolean cacheInProgress;
		synchronized (this) {
			mNumBlocks = _numBlocks;
			cacheInProgress = _cacheInProgress;
			if (incrConsumers)
				maxConsumptionCount++;
			if (cacheInProgress) {
				int newLen = _subscribers == null ? 1 : _subscribers.length + 1;
				Runnable[] newSubscribers = new Runnable[newLen];

				if(newLen > 1)
					System.arraycopy(_subscribers, 0, newSubscribers, 0, newLen - 1);

				newSubscribers[newLen - 1] = subscriber;
				_subscribers = newSubscribers;
			}
		}

		for (int i = 0; i < mNumBlocks; i++)
			subscriber.run();

		if (!cacheInProgress)
			subscriber.run(); // To fetch the NO_MORE_TASK element
	}

	/**
	 * Artificially increase subscriber count.
	 * Only use if certain blocks are accessed more than once.
	 */
	public synchronized void incrSubscriberCount(int count) {
		maxConsumptionCount += count;
	}

	/**
	 * Artificially increase the processing count of a block.
	 */
	public synchronized void incrProcessingCount(int i, int count) {
		_consumptionCounts.set(i, _consumptionCounts.getInt(i)+count);

		if (deletable)
			tryDeleteBlock(i);
	}

	/**
	 * Force pins blocks in the cache to not be subject to block deletion.
	 */
	public synchronized void pinStream() {
		cachePins++;
	}

	/**
	 * Unpins the stream, allowing blocks to be deleted from cache.
	 */
	public synchronized void unpinStream() {
		cachePins--;

		if (cachePins == 0) {
			for (int i = 0; i < _consumptionCounts.size(); i++)
				tryDeleteBlock(i);
		}
	}
}
