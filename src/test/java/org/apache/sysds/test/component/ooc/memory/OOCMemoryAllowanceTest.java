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

package org.apache.sysds.test.component.ooc.memory;

import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.instructions.ooc.OOCInstruction;
import org.apache.sysds.runtime.instructions.ooc.OOCStream;
import org.apache.sysds.runtime.instructions.ooc.SubscribableTaskQueue;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.RightScalarOperator;
import org.apache.sysds.runtime.ooc.cache.OOCCacheManager;
import org.apache.sysds.runtime.ooc.memory.CachedAllowance;
import org.apache.sysds.runtime.ooc.memory.GlobalMemoryBroker;
import org.apache.sysds.runtime.ooc.memory.InMemoryQueueCallback;
import org.apache.sysds.runtime.ooc.memory.MemoryAllowance;
import org.apache.sysds.runtime.ooc.memory.MemoryBroker;
import org.apache.sysds.runtime.ooc.memory.SyncMemoryAllowance;
import org.junit.Assert;
import org.junit.Test;
import scala.Tuple3;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BiFunction;
import java.util.function.Function;

public class OOCMemoryAllowanceTest {
	private static final int TILES = 20000;

	@Test
	public void testOptimal() {
		test(true, 0, 1);
	}

	@Test
	public void testWorstCase() {
		test(false, 0, 1);
	}

	public void test(boolean optimal, int nWarmup, int nMeasure) {
		//DMLScript.OOC_STATISTICS = true;
		long millis;
		for(int i = 0; i < nWarmup; i++) {
			testNew(optimal);
		}
		millis = 0;
		for(int i = 0; i < nMeasure; i++) {
			millis += testNew(optimal);
		}
		//System.out.println("New: " + millis + "ms");
		//System.out.println(Statistics.displayOOCEvictionStats());
		OOCCacheManager.reset();

		/*for(int i = 0; i < 10; i++) {
			testOld(optimal);
		}
		millis = 0;
		for(int i = 0; i < 10; i++) {
			millis += testOld(optimal);
		}
		//System.out.println("Old: " + millis + "ms");
		//System.out.println(Statistics.displayOOCEvictionStats());
		OOCCacheManager.reset();*/
	}

	public long testNew(boolean optimal) {
		// We emulate the expression (A + 2) + B with limited memory
		MemoryBroker parentBroker = new GlobalMemoryBroker(500000000L);
		CoordinatedBroker broker = new CoordinatedBroker(parentBroker);
		TestInstruction test = new TestInstruction();

		MemoryAllowance leftAllowance = new SyncMemoryAllowance(broker);
		MemoryAllowance rightAllowance = new SyncMemoryAllowance(broker);
		MemoryAllowance joinAllowance = new SyncMemoryAllowance(broker);
		CachedAllowance cache = new CachedAllowance(broker);

		OOCStream<Integer> leftStream = new SubscribableTaskQueue<>();
		OOCStream<Integer> rightStream = new SubscribableTaskQueue<>();
		OOCStream<InMemoryQueueCallback> outStream = new SubscribableTaskQueue<>();

		long startMillis = System.currentTimeMillis();

		// Left producer reservation thread
		new Thread(() -> {
			for(int i = 0; i < TILES; i++) {
				leftAllowance.reserveBlocking(8 * 1000 + /* Working memory */ 8 * 1000);
				leftStream.enqueue(i);
			}
			leftStream.closeInput();
		}).start();

		// Right producer reservation thread
		new Thread(() -> {
			for(int i = 0; i < TILES; i++) {
				rightAllowance.reserveBlocking(8 * 1000); // Needs no working memory
				if(optimal)
					rightStream.enqueue(i);
				else
					rightStream.enqueue(TILES-i-1);
			}
			rightStream.closeInput();
		}).start();

		OOCStream<InMemoryQueueCallback> leftStreamOut = new SubscribableTaskQueue<>();
		OOCStream<InMemoryQueueCallback> leftStreamOutOut = new SubscribableTaskQueue<>();
		OOCStream<InMemoryQueueCallback> rightStreamOut = new SubscribableTaskQueue<>();

		test.map(leftStream, leftStreamOut, i -> {
			var imv = new IndexedMatrixValue(new MatrixIndexes(i.longValue(), 1L), new MatrixBlock(1000, 1, 5.0));
			return new InMemoryQueueCallback(imv, null, leftAllowance, 8 * 1000);
		});
		test.map(leftStreamOut, leftStreamOutOut, cb -> {
			try(cb) {
				var imv = new IndexedMatrixValue(cb.get().getIndexes(), cb.get().getValue()
					.scalarOperations(new RightScalarOperator(Plus.getPlusFnObject(), 2.0), new MatrixBlock()));
				return new InMemoryQueueCallback(imv, null, leftAllowance, 8 * 1000);
			}
		});
		test.map(rightStream, rightStreamOut, i -> {
			var imv = new IndexedMatrixValue(new MatrixIndexes(i.longValue(), 1L), new MatrixBlock(1000, 1, 3.0));
			return new InMemoryQueueCallback(imv, null, rightAllowance, 8 * 1000);
		});

		test.join(leftStreamOutOut, rightStreamOut, outStream, () -> joinAllowance.reserveBlocking(8 * 1000), cache,
			(l, r) -> {
				var imv = new IndexedMatrixValue(l.getIndexes(), ((MatrixBlock)l.getValue()).binaryOperations(new BinaryOperator(
					Plus.getPlusFnObject()), r.getValue()));
				return new InMemoryQueueCallback(imv, null, joinAllowance, 8 * 1000);
		});

		CompletableFuture<Void> future = new CompletableFuture<>();
		AtomicInteger ctr = new AtomicInteger();
		outStream.setSubscriber(cb -> {
			try {
				if(cb.isEos()) {
					future.complete(null);
					return;
				}
				InMemoryQueueCallback inner = cb.get();
				try(cb; inner) {
					ctr.incrementAndGet();
					double checksum =((MatrixBlock)inner.get().getValue()).sum();
					if(checksum < 10000.0 - 1e-9 || checksum > 10000.0 + 1e-9)
						future.completeExceptionally(new AssertionError("Wrong checksum: " + checksum));
					//System.out.println(cb.get().get().getIndexes());
				}
			}
			catch(Exception e) {
				future.completeExceptionally(e);
			}
		});
		future.join();

		Assert.assertEquals(TILES, ctr.get());
		return System.currentTimeMillis() - startMillis;
	}

	public long testOld(boolean optimal) {
		// We emulate the expression (A + 2) + B with limited memory
		TestInstruction test = new TestInstruction();

		OOCStream<Integer> leftStream = new SubscribableTaskQueue<>();
		OOCStream<Integer> rightStream = new SubscribableTaskQueue<>();
		OOCStream<IndexedMatrixValue> outStream = new SubscribableTaskQueue<>();

		long startMillis = System.currentTimeMillis();

		// Left producer reservation thread
		new Thread(() -> {
			for(int i = 0; i < TILES; i++) {
				leftStream.enqueue(i);
			}
			leftStream.closeInput();
		}).start();

		// Right producer reservation thread
		new Thread(() -> {
			for(int i = 0; i < TILES; i++) {
				if(optimal)
					rightStream.enqueue(i);
				else
					rightStream.enqueue(TILES-i-1);
			}
			rightStream.closeInput();
		}).start();

		OOCStream<IndexedMatrixValue> leftStreamOut = new SubscribableTaskQueue<>();
		OOCStream<IndexedMatrixValue> leftStreamOutOut = new SubscribableTaskQueue<>();
		OOCStream<IndexedMatrixValue> rightStreamOut = new SubscribableTaskQueue<>();

		test.map(leftStream, leftStreamOut, i -> {
			var imv = new IndexedMatrixValue(new MatrixIndexes(i.longValue(), 1L), new MatrixBlock(1000, 1, 5.0));
			return imv;
		});
		test.map(leftStreamOut, leftStreamOutOut, v -> {
			var imv = new IndexedMatrixValue(v.getIndexes(), v.getValue()
				.scalarOperations(new RightScalarOperator(Plus.getPlusFnObject(), 2.0), new MatrixBlock()));
			return imv;
		});
		test.map(rightStream, rightStreamOut, i -> {
			var imv = new IndexedMatrixValue(new MatrixIndexes(i.longValue(), 1L), new MatrixBlock(1000, 1, 3.0));
			return imv;
		});

		test.joinOOC(leftStreamOutOut, rightStreamOut, outStream,
			(l, r) -> {
				var imv = new IndexedMatrixValue(l.getIndexes(), ((MatrixBlock)l.getValue()).binaryOperations(new BinaryOperator(
					Plus.getPlusFnObject()), r.getValue()));
				return imv;
			});

		CompletableFuture<Void> future = new CompletableFuture<>();
		outStream.setSubscriber(cb -> {
			try {
				if(cb.isEos()) {
					future.complete(null);
					return;
				}
				try(cb) {
					//System.out.println(cb.get().getIndexes());
				}
			}
			catch(Exception e) {
				e.printStackTrace();
			}
		});
		future.join();
		return System.currentTimeMillis() - startMillis;
	}

	static class TestInstruction extends OOCInstruction {
		protected TestInstruction() {
			super(null, "test", "test");
		}

		@Override
		public void processInstruction(ExecutionContext ec) {
		}

		public <T, R> CompletableFuture<Void> map(OOCStream<T> qIn, OOCStream<R> qOut, Function<T, R> mapper) {
			return mapOOC(qIn, qOut, mapper);
		}

		public CompletableFuture<Void> joinOOC(OOCStream<IndexedMatrixValue> l, OOCStream<IndexedMatrixValue> r,
			OOCStream<IndexedMatrixValue> out, BiFunction<IndexedMatrixValue, IndexedMatrixValue, IndexedMatrixValue> joinFn) {

			return super.joinOOC(l, r, out, joinFn, IndexedMatrixValue::getIndexes);
		}

		public CompletableFuture<Void> join(OOCStream<InMemoryQueueCallback> l, OOCStream<InMemoryQueueCallback> r,
			OOCStream<InMemoryQueueCallback> out, Runnable memoryReserver, CachedAllowance cache,
			BiFunction<IndexedMatrixValue, IndexedMatrixValue, InMemoryQueueCallback> joinFn) {

			OOCStream<Tuple3<OOCStream.QueueCallback<IndexedMatrixValue>, OOCStream.QueueCallback<IndexedMatrixValue>, Integer>> intermediate = createWritableStream();

			new Thread(() -> {
				InMemoryQueueCallback next;
				IndexedMatrixValue nextValue;
				boolean nextLeft = true;
				AtomicInteger pendingRequests = new AtomicInteger(1);

				while((next = (nextLeft ? l : r).dequeue()) != null) {
					try {
						nextValue = next.get();
						int idx = (int) nextValue.getIndexes().getRowIndex();
						var future = cache.get(idx);
						if(future.isDone()) {
							var cb = future.getNow(null);
							if(cb == null) {
								cache.handover(next, idx);
							}
							else {
								try(cb) {
									memoryReserver.run(); // reserve memory for future pipeline
									intermediate.enqueue(nextLeft ? new Tuple3<>(next.keepOpen(), cb.keepOpen(), idx) :
										new Tuple3<>(cb.keepOpen(), next.keepOpen(), idx));
								}
							}
						}
						else {
							pendingRequests.incrementAndGet();
							final var pinned = next.keepOpen();
							final boolean isLeft = nextLeft;
							future.thenAccept(cb -> {
								try(cb; pinned) {
									intermediate.enqueue(
										isLeft ? new Tuple3<>(pinned.keepOpen(), cb.keepOpen(), idx) :
											new Tuple3<>(cb.keepOpen(), pinned.keepOpen(), idx));
								}
								if(pendingRequests.decrementAndGet() == 0)
									intermediate.closeInput();
							});
						}

						nextLeft = !nextLeft;
					}
					finally {
						next.close();
					}
				}

				if(pendingRequests.decrementAndGet() == 0)
					intermediate.closeInput();
			}).start();

			return mapOOC(intermediate, out, t -> {
				var qL = t._1();
				var qR = t._2();
				try(qL; qR) {
					return joinFn.apply(qL.get(), qR.get());
				}
				finally {
					cache.clear(t._3());
				}
			});
		}
	}

	static class CoordinatedBroker extends SyncMemoryAllowance implements MemoryBroker {
		private final List<MemoryAllowance> _children;
		private final Map<MemoryAllowance, Long> _credits;
		private record TargetUpdate(MemoryAllowance allowance, long target) {}

		CoordinatedBroker(MemoryBroker parentBroker) {
			super(parentBroker);
			_children = new ArrayList<>();
			_credits = new IdentityHashMap<>();
		}

		@Override
		public void attachAllowance(MemoryAllowance allowance) {
			List<TargetUpdate> updates;
			synchronized(this) {
				_children.add(allowance);
				_credits.put(allowance, 0L);
				updates = rebalanceTargetsLocked();
			}
			applyTargetUpdates(updates);
		}

		@Override
		public long requestMemory(MemoryAllowance allowance, long minSize, long maxSize) {
			if(!_credits.containsKey(allowance))
				throw new UnsupportedOperationException("Allowance is not attached to CoordinatedBroker.");
			List<TargetUpdate> updates;
			long granted;
			synchronized(this) {
				granted = requestGrantLocked(allowance, minSize);
				updates = rebalanceTargetsLocked();
			}
			applyTargetUpdates(updates);
			return granted;
		}

		@Override
		public void freeMemory(MemoryAllowance allowance, long freedMemory) {
			if(!_credits.containsKey(allowance))
				throw new UnsupportedOperationException("Allowance is not attached to CoordinatedBroker.");
			if(freedMemory <= 0)
				return;
			List<TargetUpdate> updates;
			synchronized(this) {
				release(freedMemory);
				updates = rebalanceTargetsLocked();
			}
			applyTargetUpdates(updates);
		}

		@Override
		public void shutdownAllowance(MemoryAllowance allowance) {
			if(!_credits.containsKey(allowance))
				throw new UnsupportedOperationException("Allowance is not attached to CoordinatedBroker.");
			List<TargetUpdate> updates;
			synchronized(this) {
				updates = rebalanceTargetsLocked();
			}
			applyTargetUpdates(updates);
		}

		@Override
		public void destroyAllowance(MemoryAllowance allowance, long freedMemory) {
			if(!_credits.containsKey(allowance))
				throw new UnsupportedOperationException("Allowance is not attached to CoordinatedBroker.");
			List<TargetUpdate> updates;
			synchronized(this) {
				_children.remove(allowance);
				_credits.remove(allowance);
				if(freedMemory > 0)
					release(freedMemory);
				updates = rebalanceTargetsLocked();
			}
			applyTargetUpdates(updates);
		}

		private long requestGrantLocked(MemoryAllowance requester, long minSize) {
			int n = _children.size();
			if(n == 0)
				return 0;
			long credit = _credits.getOrDefault(requester, 0L);
			if(credit >= minSize) {
				_credits.put(requester, credit - minSize);
				return minSize;
			}

			long granted = credit;
			long missing = minSize - granted;
			long total = n * missing;
			if(!tryReserve(total))
				return 0;
			_credits.put(requester, 0L);
			for(MemoryAllowance child : _children) {
				if(child == requester)
					continue;
				_credits.put(child, _credits.getOrDefault(child, 0L) + missing);
			}
			return minSize;
		}

		private List<TargetUpdate> rebalanceTargetsLocked() {
			List<TargetUpdate> updates = new ArrayList<>(_children.size());
			long target = getTargetMemory();
			int n = _children.size();
			long share = n == 0 ? 0 : target / n;
			for(MemoryAllowance child : _children)
				updates.add(new TargetUpdate(child, share));
			return updates;
		}

		private static void applyTargetUpdates(List<TargetUpdate> updates) {
			for(TargetUpdate update : updates)
				update.allowance.setTargetMemory(update.target);
		}
	}
}
