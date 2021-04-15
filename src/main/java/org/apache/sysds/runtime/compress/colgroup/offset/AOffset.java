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
package org.apache.sysds.runtime.compress.colgroup.offset;

import java.io.DataOutput;
import java.io.IOException;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

/**
 * Offset list encoder interface.
 * 
 * It is assumed that the input is in sorted order, all values are positive and there are no duplicates.
 * 
 * The no duplicate is important since 0 values are exploited to encode an offset of max representable value + 1. This
 * gives the ability to encode data, where the offsets are greater than the available highest value that can be
 * represented size.
 */
public abstract class AOffset {

	protected static final Log LOG = LogFactory.getLog(AOffset.class.getName());

	/**
	 * Get an iterator of the offsets.
	 * 
	 * @return AIterator that iterate through index and dictionary offset values.
	 */
	public abstract AIterator getIterator();

	/**
	 * Write the offsets to disk.
	 * 
	 * If you implement another remember to write the ordinal of the new type to disk as well and add it to the
	 * OffsetFactory.
	 * 
	 * @param out The output to write to
	 * @throws IOException Exception that happens if the IO fails to write.
	 */
	public abstract void write(DataOutput out) throws IOException;

	/**
	 * Get the in memory size of the Offset object
	 * 
	 * @return In memory size as a long.
	 */
	public abstract long getInMemorySize();

	/**
	 * Remember to include the ordinal of the type of offset list.
	 * 
	 * @return the size on disk as a long.
	 */
	public abstract long getExactSizeOnDisk();

	/**
	 * Get the number of contained elements, This method iterate the entire offset list, so it is not constant lookup.
	 * 
	 * @return The number of indexes.
	 */
	public abstract int getSize();

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		AIterator i = getIterator();
		sb.append(this.getClass().getSimpleName());
		sb.append(" [");
		if(i.hasNext())
			sb.append(i.valueAndIncrement());

		while(i.hasNext())
			sb.append(", " + i.valueAndIncrement());
		sb.append("]");
		return sb.toString();
	}
}
