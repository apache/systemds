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

package org.apache.sysds.runtime.frame.data.iterators;

import java.util.Iterator;

import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.frame.data.FrameBlock;

/**
 * Factory pattern for construction of rowIterators of the FrameBlock.
 */
public interface IteratorFactory {

	/**
	 * Get a row iterator over the frame where all fields are encoded as strings independent of their value types.
	 *
	 * @param fb The frame to iterate through
	 * @return string array iterator
	 */
	public static Iterator<String[]> getStringRowIterator(FrameBlock fb) {
		return new StringRowIterator(fb, 0, fb.getNumRows());
	}

	/**
	 * Get a row iterator over the frame where all selected fields are encoded as strings independent of their value
	 * types.
	 *
	 * @param fb   The frame to iterate through
	 * @param cols column selection, 1-based
	 * @return string array iterator
	 */
	public static Iterator<String[]> getStringRowIterator(FrameBlock fb, int[] cols) {
		return new StringRowIterator(fb, 0, fb.getNumRows(), cols);
	}

	/**
	 * Get a row iterator over the frame where all selected fields are encoded as strings independent of their value
	 * types.
	 *
	 * @param fb    The frame to iterate through
	 * @param colID column selection, 1-based
	 * @return string array iterator
	 */
	public static Iterator<String[]> getStringRowIterator(FrameBlock fb, int colID) {
		return new StringRowIterator(fb, 0, fb.getNumRows(), new int[] {colID});
	}

	/**
	 * Get a row iterator over the frame where all fields are encoded as strings independent of their value types.
	 *
	 * @param fb The frame to iterate through
	 * @param rl lower row index
	 * @param ru upper row index
	 * @return string array iterator
	 */
	public static Iterator<String[]> getStringRowIterator(FrameBlock fb, int rl, int ru) {
		return new StringRowIterator(fb, rl, ru);
	}

	/**
	 * Get a row iterator over the frame where all selected fields are encoded as strings independent of their value
	 * types.
	 *
	 * @param fb   The frame to iterate through
	 * @param rl   lower row index
	 * @param ru   upper row index
	 * @param cols column selection, 1-based
	 * @return string array iterator
	 */
	public static Iterator<String[]> getStringRowIterator(FrameBlock fb, int rl, int ru, int[] cols) {
		return new StringRowIterator(fb, rl, ru, cols);
	}

	/**
	 * Get a row iterator over the frame where all selected fields are encoded as strings independent of their value
	 * types.
	 *
	 * @param fb    The frame to iterate through
	 * @param rl    lower row index
	 * @param ru    upper row index
	 * @param colID columnID, 1-based
	 * @return string array iterator
	 */
	public static Iterator<String[]> getStringRowIterator(FrameBlock fb, int rl, int ru, int colID) {
		return new StringRowIterator(fb, rl, ru, new int[] {colID});
	}

	/**
	 * Get a row iterator over the frame where all fields are encoded as boxed objects according to their value types.
	 *
	 * @param fb The frame to iterate through
	 * @return object array iterator
	 */
	public static Iterator<Object[]> getObjectRowIterator(FrameBlock fb) {
		return new ObjectRowIterator(fb, 0, fb.getNumRows());
	}

	/**
	 * Get a row iterator over the frame where all fields are encoded as boxed objects according to the value types of
	 * the provided target schema.
	 *
	 * @param fb     The frame to iterate through
	 * @param schema target schema of objects
	 * @return object array iterator
	 */
	public static Iterator<Object[]> getObjectRowIterator(FrameBlock fb, ValueType[] schema) {
		return new ObjectRowIterator(fb, 0, fb.getNumRows(), schema);
	}

	/**
	 * Get a row iterator over the frame where all selected fields are encoded as boxed objects according to their value
	 * types.
	 *
	 * @param fb   The frame to iterate through
	 * @param cols column selection, 1-based
	 * @return object array iterator
	 */
	public static Iterator<Object[]> getObjectRowIterator(FrameBlock fb, int[] cols) {
		return new ObjectRowIterator(fb, 0, fb.getNumRows(), cols);
	}

	/**
	 * Get a row iterator over the frame where all fields are encoded as boxed objects according to their value types.
	 *
	 * @param fb The frame to iterate through
	 * @param rl lower row index
	 * @param ru upper row index
	 * @return object array iterator
	 */
	public static Iterator<Object[]> getObjectRowIterator(FrameBlock fb, int rl, int ru) {
		return new ObjectRowIterator(fb, rl, ru);
	}

	/**
	 * Get a row iterator over the frame where all selected fields are encoded as boxed objects according to their value
	 * types.
	 *
	 * @param fb   The frame to iterate through
	 * @param rl   lower row index
	 * @param ru   upper row index
	 * @param cols column selection, 1-based
	 * @return object array iterator
	 */
	public static Iterator<Object[]> getObjectRowIterator(FrameBlock fb, int rl, int ru, int[] cols) {
		return new ObjectRowIterator(fb, rl, ru, cols);
	}

}
