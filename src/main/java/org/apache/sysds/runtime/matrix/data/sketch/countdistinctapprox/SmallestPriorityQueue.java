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

package org.apache.sysds.runtime.matrix.data.sketch.countdistinctapprox;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import java.util.Collections;
import java.util.HashSet;
import java.util.PriorityQueue;
import java.util.Set;

/**
 * Deceiving name, but is used to contain the k smallest values inserted.
 *
 * TODO: Replace Standard Java Set and Priority Queue with optimized versions.
 */
public class SmallestPriorityQueue {
	private static final Log LOG = LogFactory.getLog(SmallestPriorityQueue.class.getName());

	private Set<Double> containedSet;
	private PriorityQueue<Double> smallestHashes;
	private int k;

	public SmallestPriorityQueue(int k) {
		smallestHashes = new PriorityQueue<>(k, Collections.reverseOrder());
		containedSet = new HashSet<>(1);
		this.k = k;
	}

	public void add(double v) {
		if(!containedSet.contains(v)) {
			if(smallestHashes.size() < k) {
				smallestHashes.add(v);
				containedSet.add(v);
			}
			else if(v < smallestHashes.peek()) {
				LOG.trace(smallestHashes.peek() + " -- " + v);
				smallestHashes.add(v);
				containedSet.add(v);
				double largest = smallestHashes.poll();
				containedSet.remove(largest);
			}
		}
	}

	public int size() {
		return smallestHashes.size();
	}

	public double peek() {
		return smallestHashes.peek();
	}

	public double poll() {
		return smallestHashes.poll();
	}

	public boolean isEmpty() {
		return this.size() == 0;
	}

	public void clear() {
		this.containedSet.clear();
		this.smallestHashes.clear();
	}

	@Override
	public String toString() {
		return smallestHashes.toString();
	}
}
