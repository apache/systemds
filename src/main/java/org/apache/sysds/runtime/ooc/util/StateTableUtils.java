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

package org.apache.sysds.runtime.ooc.util;

import org.apache.sysds.runtime.instructions.ooc.OOCStream;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.ooc.cache.OOCFuture;
import org.apache.sysds.runtime.ooc.memory.InMemoryQueueCallback;
import org.apache.sysds.runtime.ooc.memory.ManagedPayload;
import org.apache.sysds.runtime.ooc.memory.MemoryAllowance;
import org.apache.sysds.runtime.ooc.store.MaterializedCallback;
import org.apache.sysds.runtime.ooc.store.StateTable;
import org.apache.sysds.runtime.ooc.store.StoreLease;

public final class StateTableUtils {
	public static OOCFuture<Match> putOrTake(StateTable<IndexedMatrixValue> table, int slot,
		OOCStream.QueueCallback<IndexedMatrixValue> tile, MemoryAllowance allowance) {
		if(tile instanceof MaterializedCallback pinned && pinned.pinnedEntry() != null)
			return putReferenceOrTake(table, slot, pinned, allowance);
		ManagedPayload<IndexedMatrixValue> payload;
		if(tile instanceof InMemoryQueueCallback managed && managed.getManagedBytes() > 0) {
			payload = managed.extractManagedPayload();
			managed.close();
		}
		else {
			IndexedMatrixValue value = tile.get();
			long bytes = ((MatrixBlock) value.getValue()).getExactSerializedSize();
			allowance.reserveBlocking(bytes);
			payload = new ManagedPayload<>(value, bytes, allowance);
			tile.close();
		}
		OOCFuture<Match> result = new OOCFuture<>();
		OOCFuture<StoreLease<IndexedMatrixValue>> matched;
		try {
			matched = table.putOrTake(slot, payload, allowance);
		}
		catch(RuntimeException ex) {
			payload.release();
			return OOCFuture.failed(ex);
		}
		matched.whenComplete((lease, error) -> {
			if(error != null) {
				payload.release();
				result.completeExceptionally(error);
			}
			else if(lease == null)
				result.complete(null);
			else
				result.complete(new Match(new MaterializedCallback(new StoreLease<>(payload.value(), payload::release)),
					new MaterializedCallback(lease)));
		});
		return result;
	}

	private static OOCFuture<Match> putReferenceOrTake(StateTable<IndexedMatrixValue> table, int slot,
		MaterializedCallback pinned, MemoryAllowance allowance) {
		OOCFuture<Match> result = new OOCFuture<>();
		OOCFuture<StoreLease<IndexedMatrixValue>> matched;
		try {
			matched = table.putReferenceOrTake(slot, pinned.pinnedEntry(), allowance);
		}
		catch(RuntimeException ex) {
			pinned.close();
			return OOCFuture.failed(ex);
		}
		matched.whenComplete((lease, error) -> {
			if(error != null) {
				pinned.close();
				result.completeExceptionally(error);
			}
			else if(lease == null) {
				pinned.close();
				result.complete(null);
			}
			else
				result.complete(new Match(pinned, new MaterializedCallback(lease)));
		});
		return result;
	}

	public record Match(OOCStream.QueueCallback<IndexedMatrixValue> left,
		OOCStream.QueueCallback<IndexedMatrixValue> right) {
	}
}
