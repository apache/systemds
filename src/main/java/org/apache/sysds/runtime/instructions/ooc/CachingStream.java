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

import org.apache.commons.collections4.BidiMap;
import org.apache.commons.collections4.bidimap.DualHashBidiMap;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.controlprogram.parfor.LocalTaskQueue;
import org.apache.sysds.runtime.controlprogram.parfor.util.IDSequence;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.ooc.cache.BlockKey;
import org.apache.sysds.runtime.ooc.cache.OOCIOHandler;
import org.apache.sysds.runtime.ooc.cache.OOCCacheManager;
import org.apache.sysds.runtime.ooc.stream.SourceOOCStream;
import org.apache.sysds.runtime.ooc.stream.message.OOCGetStreamTypeMessage;
import org.apache.sysds.runtime.ooc.stream.message.OOCStreamMessage;
import org.apache.sysds.runtime.util.IndexRange;
import shaded.parquet.it.unimi.dsi.fastutil.ints.IntArrayList;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.ExecutionException;
import java.util.function.BiFunction;
import java.util.function.Consumer;

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
	private final IntArrayList _consumerConsumptionCounts = new IntArrayList();

	// stream identifier
	private final long _streamId;

	// block counter
	private int _numBlocks = 0;

	private Consumer<OOCStream.QueueCallback<IndexedMatrixValue>>[] _subscribers;
	private CopyOnWriteArrayList<Consumer<OOCStreamMessage>> _downstreamRelays;

	// state flags
	private boolean _cacheInProgress = true; // caching in progress, in the first pass.
	private BidiMap<MatrixIndexes, Integer> _index;

	private DMLRuntimeException _failure;

	private boolean _deletable = false;
	private int _maxConsumptionCount = 0;
	private String _watchdogId = null;

	public CachingStream(OOCStream<IndexedMatrixValue> source) {
		this(source, _streamSeq.getNextID());
	}

	public CachingStream(OOCStream<IndexedMatrixValue> source, long streamId) {
		_source = source;
		_source.setDownstreamMessageRelay(this::messageDownstream);
		_streamId = streamId;
		if (OOCWatchdog.WATCH) {
			_watchdogId = "CS-" + hashCode();
			// Capture a short context to help identify origin
			OOCWatchdog.registerOpen(_watchdogId, "CachingStream@" + hashCode(), getCtxMsg(), this);
		}
		_downstreamRelays = null;
		source.setSubscriber(tmp -> {
			try (tmp) {
				final IndexedMatrixValue task = tmp.get();
				int blk;
				Consumer<OOCStream.QueueCallback<IndexedMatrixValue>>[] mSubscribers;
				OOCStream.QueueCallback<IndexedMatrixValue> mCallback = null;

				synchronized (this) {
					mSubscribers = _subscribers;
					if(task != LocalTaskQueue.NO_MORE_TASKS) {
						if(!_cacheInProgress)
							throw new DMLRuntimeException("Stream is closed");
						OOCIOHandler.SourceBlockDescriptor descriptor = null;
						if(_source instanceof SourceOOCStream src) {
							descriptor = src.getDescriptor(task.getIndexes());
						}
						if(descriptor == null) {
							if(mSubscribers == null || mSubscribers.length == 0)
								OOCCacheManager.put(_streamId, _numBlocks, task);
							else
								mCallback = OOCCacheManager.putAndPin(_streamId, _numBlocks, task);
						}
						else {
							if(mSubscribers == null || mSubscribers.length == 0)
								OOCCacheManager.putSourceBacked(_streamId, _numBlocks, task, descriptor);
							else
								mCallback = OOCCacheManager.putAndPinSourceBacked(_streamId, _numBlocks, task,
									descriptor);
						}
						if(_index != null)
							_index.put(task.getIndexes(), _numBlocks);
						blk = _numBlocks;
						_numBlocks++;
						_consumptionCounts.add(0);
						notifyAll();
					}
					else {
						_cacheInProgress = false; // caching is complete
						try {
							validateBlockCountOnClose();
						}
						catch(Exception e) {
							_failure = e instanceof DMLRuntimeException ? (DMLRuntimeException) e : new DMLRuntimeException(e);
						}
						if (OOCWatchdog.WATCH)
							OOCWatchdog.registerClose(_watchdogId);
						notifyAll();
						blk = -1;
					}
				}

				if(mSubscribers != null && mSubscribers.length > 0) {
					final OOCStream.QueueCallback<IndexedMatrixValue> finalCallback = mCallback;
					try(finalCallback) {
						if(blk != -1) {
							for(int i = 0; i < mSubscribers.length; i++) {
								OOCStream.QueueCallback<IndexedMatrixValue> localCallback = finalCallback.keepOpen();
								try(localCallback) {
									mSubscribers[i].accept(localCallback);
								}
								if(onConsumed(blk, i))
									mSubscribers[i].accept(OOCStream.eos(_failure));
							}
						}
						else {
							OOCStream.QueueCallback<IndexedMatrixValue> cb = OOCStream.eos(_failure);
							for(int i = 0; i < mSubscribers.length; i++) {
								if(onNoMoreTasks(i))
									mSubscribers[i].accept(cb);
							}
						}
					}
				}
			} catch (DMLRuntimeException e) {
				// Propagate failure to subscribers
				_failure = e;
				synchronized (this) {
					notifyAll();
				}

				Consumer<OOCStream.QueueCallback<IndexedMatrixValue>>[] mSubscribers = _subscribers;
				OOCStream.QueueCallback<IndexedMatrixValue> err = OOCStream.eos(_failure);
				if(mSubscribers != null) {
					for(Consumer<OOCStream.QueueCallback<IndexedMatrixValue>> mSubscriber : mSubscribers) {
						try {
							mSubscriber.accept(err);
						} catch (Exception ignored) {
						}
					}
				}
			}
		});
	}

	private String getCtxMsg() {
		StackTraceElement[] st = new Exception().getStackTrace();
		// Skip the first few frames (constructor, createWritableStream, etc.)
		StringBuilder sb = new StringBuilder();
		int limit = Math.min(st.length, 7);
		for(int i = 2; i < limit; i++) {
			sb.append(st[i].getClassName()).append(".").append(st[i].getMethodName()).append(":")
				.append(st[i].getLineNumber());
			if(i < limit - 1)
				sb.append(" <- ");
		}
		return sb.toString();
	}

	public synchronized void scheduleDeletion() {
		if (_deletable)
			return; // Deletion already scheduled

		if (_cacheInProgress && _maxConsumptionCount == 0)
			throw new DMLRuntimeException("Cannot have a caching stream with no listeners");

		_deletable = true;
		for (int i = 0; i < _consumptionCounts.size(); i++) {
			tryDeleteBlock(i);
		}
	}

	public String toString() {
		return "CachingStream@" + _streamId;
	}

	private synchronized void tryDeleteBlock(int i) {
		int cnt = _consumptionCounts.getInt(i);
		if (cnt > _maxConsumptionCount)
			throw new DMLRuntimeException("Cannot have more than " + _maxConsumptionCount + " consumptions.");
		if (cnt == _maxConsumptionCount)
			OOCCacheManager.forget(_streamId, i);
	}

	private synchronized boolean onConsumed(int blockIdx, int consumerIdx) {
		int newCount = _consumptionCounts.getInt(blockIdx)+1;
		if (newCount > _maxConsumptionCount)
			throw new DMLRuntimeException("Cannot have more than " + _maxConsumptionCount + " consumptions.");
		_consumptionCounts.set(blockIdx, newCount);
		int newConsumerCount = _consumerConsumptionCounts.getInt(consumerIdx)+1;
		_consumerConsumptionCounts.set(consumerIdx, newConsumerCount);

		if (_deletable)
			tryDeleteBlock(blockIdx);

		return !_cacheInProgress && newConsumerCount == _numBlocks + 1;
	}

	private synchronized boolean onNoMoreTasks(int consumerIdx) {
		int newConsumerCount = _consumerConsumptionCounts.getInt(consumerIdx)+1;
		_consumerConsumptionCounts.set(consumerIdx, newConsumerCount);
		return !_cacheInProgress && newConsumerCount == _numBlocks + 1;
	}

	public synchronized CompletableFuture<OOCStream.QueueCallback<IndexedMatrixValue>> get(int idx) throws InterruptedException,
		ExecutionException {
		while (true) {
			if(_failure != null)
				throw _failure;
			else if(idx < _numBlocks) {
				return OOCCacheManager.requestBlock(_streamId, idx)
					.thenApply(cb -> {
						synchronized(this) {
							if(_index != null) // Ensure index is up to date
								_index.putIfAbsent(cb.get().getIndexes(), idx);

							int newCount = _consumptionCounts.getInt(idx) + 1;
							if(newCount > _maxConsumptionCount)
								throw new DMLRuntimeException("Consumer overflow! Expected: " + _maxConsumptionCount);
							_consumptionCounts.set(idx, newCount);

							if(_deletable)
								tryDeleteBlock(idx);
						}
						return cb;
					});
			}
			else if(!_cacheInProgress) {
				return CompletableFuture.completedFuture(new OOCStream.SimpleQueueCallback<>(null, _failure));
			}

			wait();
		}
	}

	public synchronized int findCachedIndex(MatrixIndexes idx) {
		return _index.get(idx);
	}

	public synchronized BlockKey peekCachedBlockKey(MatrixIndexes idx) {
		return new BlockKey(_streamId, _index.get(idx));
	}

	public synchronized OOCStream.QueueCallback<IndexedMatrixValue> findCached(MatrixIndexes idx) {
		int mIdx = _index.get(idx);
		int newCount = _consumptionCounts.getInt(mIdx)+1;
		if (newCount > _maxConsumptionCount)
			throw new DMLRuntimeException("Consumer overflow in " + _streamId + "_" + mIdx + ". Expected: " +
				_maxConsumptionCount);

		_consumptionCounts.set(mIdx, newCount);

		try {
			return OOCCacheManager.requestBlock(_streamId, mIdx).get();
		} catch (InterruptedException | ExecutionException e) {
			return new OOCStream.SimpleQueueCallback<>(null, new DMLRuntimeException(e));
		} finally {
			if (_deletable)
				tryDeleteBlock(mIdx);
		}
	}

	public void findCachedAsync(MatrixIndexes idx, Consumer<OOCStream.QueueCallback<IndexedMatrixValue>> callback) {
		int mIdx;
		synchronized(this) {
			mIdx = _index.get(idx);
			int newCount = _consumptionCounts.getInt(mIdx)+1;
			if (newCount > _maxConsumptionCount)
				throw new DMLRuntimeException("Consumer overflow in " + _streamId + "_" + mIdx + ". Expected: " +
					_maxConsumptionCount);
		}
		OOCCacheManager.requestBlock(_streamId, mIdx).whenComplete((cb, r) -> {
			try (cb) {
				synchronized(CachingStream.this) {
					int newCount = _consumptionCounts.getInt(mIdx) + 1;
					if(newCount > _maxConsumptionCount) {
						_failure = new DMLRuntimeException(
							"Consumer overflow in " + _streamId + "_" + mIdx + ". Expected: " + _maxConsumptionCount);
						cb.fail(_failure);
					}
					else
						_consumptionCounts.set(mIdx, newCount);
				}

				callback.accept(cb);
			}
		});
	}

	private void validateBlockCountOnClose() {
		DataCharacteristics dc = _source.getDataCharacteristics();
		if (dc != null && dc.dimsKnown() && dc.getBlocksize() > 0) {
			long expected = dc.getNumBlocks();
			if (expected >= 0 && _numBlocks != expected) {
				throw new DMLRuntimeException("CachingStream block count mismatch: expected "
					+ expected + " but saw " + _numBlocks + " (" + dc.getRows() + "x" + dc.getCols() + ")");
			}
		}
	}

	/**
	 * Finds a cached item asynchronously without counting it as a consumption.
	 */
	public void peekCachedAsync(MatrixIndexes idx, Consumer<OOCStream.QueueCallback<IndexedMatrixValue>> callback) {
		int mIdx;
		synchronized(this) {
			mIdx = _index.get(idx);
		}
		OOCCacheManager.requestBlock(_streamId, mIdx).whenComplete((cb, r) -> callback.accept(cb));
	}

	/**
	 * Finds a cached item without counting it as a consumption.
	 */
	public OOCStream.QueueCallback<IndexedMatrixValue> peekCached(MatrixIndexes idx) {
		int mIdx;
		synchronized(this) {
			mIdx = _index.get(idx);
		}
		try {
			return OOCCacheManager.requestBlock(_streamId, mIdx).get();
		} catch (InterruptedException | ExecutionException e) {
			return new OOCStream.SimpleQueueCallback<>(null, new DMLRuntimeException(e));
		}
	}

	public synchronized void activateIndexing() {
		if (_index == null)
			_index = new DualHashBidiMap<>();
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

	@Override
	public DataCharacteristics getDataCharacteristics() {
		return _source.getDataCharacteristics();
	}

	@Override
	public CacheableData<?> getData() {
		return _source.getData();
	}

	@Override
	public void setData(CacheableData<?> data) {
		_source.setData(data);
	}

	@Override
	public void messageUpstream(OOCStreamMessage msg) {
		if (msg.isCancelled())
			return;
		if(msg instanceof OOCGetStreamTypeMessage) {
			((OOCGetStreamTypeMessage) msg).setCachedType();
			activateIndexing();
			return;
		}

		_source.messageUpstream(msg);
	}

	@Override
	public void messageDownstream(OOCStreamMessage msg) {
		CopyOnWriteArrayList<Consumer<OOCStreamMessage>> relays = _downstreamRelays;
		if (relays != null) {
			for (Consumer<OOCStreamMessage> relay : relays) {
				if (msg.isCancelled())
					break;
				relay.accept(msg);
			}
		}
	}

	@Override
	public void setUpstreamMessageRelay(Consumer<OOCStreamMessage> relay) {
		throw new UnsupportedOperationException();
	}

	@Override
	public void setDownstreamMessageRelay(Consumer<OOCStreamMessage> relay) {
		addDownstreamMessageRelay(relay);
	}

	@Override
	public void addUpstreamMessageRelay(Consumer<OOCStreamMessage> relay) {
		throw new UnsupportedOperationException();
	}

	@Override
	public void addDownstreamMessageRelay(Consumer<OOCStreamMessage> relay) {
		if (relay == null)
			throw new IllegalArgumentException("Cannot set downstream relay to null");
		CopyOnWriteArrayList<Consumer<OOCStreamMessage>> relays = _downstreamRelays;
		if (relays == null) {
			synchronized(this) {
				if (_downstreamRelays == null)
					_downstreamRelays = new CopyOnWriteArrayList<>();
				relays = _downstreamRelays;
			}
		}
		relays.add(0, relay);
	}

	@Override
	public void clearUpstreamMessageRelays() {
		// No upstream relays supported
	}

	@Override
	public void clearDownstreamMessageRelays() {
		_downstreamRelays = null;
	}

	@Override
	public void setIXTransform(BiFunction<Boolean, IndexRange, IndexRange> transform) {
		throw new UnsupportedOperationException();
	}

	public void setSubscriber(Consumer<OOCStream.QueueCallback<IndexedMatrixValue>> subscriber, boolean incrConsumers) {
		if(_deletable)
			throw new DMLRuntimeException("Cannot register a new subscriber on " + this + " because has been flagged for deletion");
		if(_failure != null)
			throw _failure;

		int mNumBlocks;
		boolean cacheInProgress;
		int consumerIdx;
		synchronized(this) {
			mNumBlocks = _numBlocks;
			cacheInProgress = _cacheInProgress;
			consumerIdx = _consumerConsumptionCounts.size();
			_consumerConsumptionCounts.add(0);
			if(incrConsumers)
				_maxConsumptionCount++;
			if(cacheInProgress) {
				int newLen = _subscribers == null ? 1 : _subscribers.length + 1;
				Consumer<OOCStream.QueueCallback<IndexedMatrixValue>>[] newSubscribers = new Consumer[newLen];

				if(newLen > 1)
					System.arraycopy(_subscribers, 0, newSubscribers, 0, newLen - 1);

				newSubscribers[newLen - 1] = subscriber;
				_subscribers = newSubscribers;
			}
		}

		for(int i = 0; i < mNumBlocks; i++) {
			final int idx = i;
			OOCCacheManager.requestBlock(_streamId, i).whenComplete((cb, r) -> {
				if(r != null) {
					subscriber.accept(OOCStream.eos(DMLRuntimeException.of(r)));
					return;
				}
				try(cb) {
					synchronized(CachingStream.this) {
						if(_index != null)
							_index.put(cb.get().getIndexes(), idx);
					}
					subscriber.accept(cb);

					if(onConsumed(idx, consumerIdx))
						subscriber.accept(OOCStream.eos(_failure)); // NO_MORE_TASKS
				}
			});
		}

		if (!cacheInProgress && onNoMoreTasks(consumerIdx))
			subscriber.accept(OOCStream.eos(_failure)); // NO_MORE_TASKS
	}

	/**
	 * Artificially increase subscriber count.
	 * Only use if certain blocks are accessed more than once.
	 */
	public synchronized void incrSubscriberCount(int count) {
		if (_deletable)
			throw new IllegalStateException("Cannot increment the subscriber count if flagged for deletion");

		_maxConsumptionCount += count;
	}

	/**
	 * Artificially increase the processing count of a block.
	 */
	public synchronized void incrProcessingCount(int i, int count) {
		int cnt = _consumptionCounts.getInt(i)+count;
		_consumptionCounts.set(i, cnt);

		if (_deletable)
			tryDeleteBlock(i);
	}
}
