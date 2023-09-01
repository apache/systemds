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

package org.apache.sysds.runtime.compress.utils;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

public abstract class ACount<T> {
	protected static final Log LOG = LogFactory.getLog(ACount.class.getName());

	/** The current count of this element */
	public int count;
	/** The current ID of this element should be unique for the user. */
	public int id;

	public abstract ACount<T> next();

	public abstract void setNext(ACount<T> e);

	public abstract T key();

	public abstract ACount<T> get(T key);

	public abstract ACount<T> inc(T key, int c, int id);

	public ACount<T> sort() {
		Sorter<T> s = new Sorter<>();
		s.sort(this);
		return s.sorted;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(key().toString());
		sb.append("=<");
		sb.append(id);
		sb.append(",");
		sb.append(count);
		sb.append(">");
		if(next() != null) {
			sb.append(" -> ");
			sb.append(next().toString());
		}
		return sb.toString();
	}

	public static final class DArrCounts extends ACount<DblArray> {
		public final DblArray key;
		public DArrCounts next;

		public DArrCounts(DblArray key, int id) {
			this.key = new DblArray(key);
			this.id = id;
			count = 1;
		}

		public DArrCounts(DblArray key, int id, int count) {
			this.key = new DblArray(key);
			this.id = id;
			this.count = count;
		}

		@Override
		public final DArrCounts next() {
			return next;
		}

		@Override
		public final void setNext(ACount<DblArray> e) {
			next = (DArrCounts) e;
		}

		@Override
		public final DblArray key() {
			return key;
		}

		@Override
		public final ACount<DblArray> get(DblArray key) {
			DArrCounts e = this;
			boolean eq = e.key.equals(key);
			while(e.next != null && !eq) {
				e = e.next;
				eq = e.key.equals(key);
			}
			return eq ? e : null;
		}

		@Override
		public final DArrCounts inc(final DblArray key, final int c, final int id) {
			// once this method jit compile it becomse 2x faster.
			DArrCounts e = this;
			if(e.key.equals(key)) {
				e.count += c;
				return e;
			}
			while(e.next != null) {
				e = e.next;
				if(e.key.equals(key)) {
					e.count += c;
					return e;
				}
			}

			e.next = new DArrCounts(key, id, c);
			return e.next;
		}

	}

	public static final class DCounts extends ACount<Double> {
		final public double key;
		public DCounts next;

		public DCounts(double key, int id) {
			this.key = key;
			this.id = id;
			this.count = 1;
		}

		public DCounts(double key, int id, int c) {
			this.key = key;
			this.id = id;
			this.count = c;
		}

		@Override
		public final DCounts next() {
			return next;
		}

		@Override
		public final void setNext(ACount<Double> e) {
			next = (DCounts) e;
		}

		@Override
		public final Double key() {
			return key;
		}

		@Override
		public DCounts sort() {
			return (DCounts) super.sort();
		}

		@Override
		public ACount<Double> get(Double key) {
			DCounts e = this;
			while(e != null && !eq(key, e.key))
				e = e.next;
			return e;
		}

		@Override
		public DCounts inc(Double key, int c, int id) {
			DCounts e = this;
			while(e.next != null && !eq(key, e.key)) {
				e = e.next;
			}

			if(eq(key, e.key)) {
				e.count += c;
				return e;
			}
			else { // e.next is null;
				e.next = new DCounts(key, id, c);
				return e.next;
			}
		}

		private static boolean eq(double a, double b) {
			long al = Double.doubleToRawLongBits(a);
			long bl = Double.doubleToRawLongBits(b);
			return al == bl;
		}

		public static final int hashIndex(double key) {
			final long bits = Double.doubleToLongBits(key);
			return Math.abs((int) (bits ^ (bits >>> 32)));
		}
	}

	private static class Sorter<T> {
		ACount<T> sorted = null;

		private void sort(ACount<T> head) {
			ACount<T> current = head;
			ACount<T> prev = null;
			ACount<T> next = null;
			// reverse
			while(current != null) {
				next = current.next();
				current.setNext(prev);
				prev = current;
				current = next;
			}
			// insert
			while(prev != null) {
				next = prev.next();
				sortedInsert(prev);
				prev = next;
			}

		}

		private void sortedInsert(ACount<T> n) {
			if(sorted == null || sorted.count < n.count) {
				n.setNext(sorted);
				sorted = n;
			}
			else {
				ACount<T> c = sorted;
				while(c.next() != null && c.next().count > n.count)
					c = c.next();
				n.setNext(c.next());
				c.setNext(n);
			}
		}

	}
}
