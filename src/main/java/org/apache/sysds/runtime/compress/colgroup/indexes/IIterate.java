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

package org.apache.sysds.runtime.compress.colgroup.indexes;

/**
 * Class to iterate through the columns of a IColIndex.
 * 
 * When initialized it should be at index -1 and then at the call to next you get the first value
 */
public interface IIterate {
	/**
	 * Get next index
	 * 
	 * @return the index.
	 */
	public int next();

	/**
	 * Get if the index has a next index.
	 * 
	 * @return the next index.
	 */
	public boolean hasNext();

	/**
	 * Get current value
	 * 
	 * @return the value pointing at.
	 */
	public int v();

	/**
	 * Get current index
	 * 
	 * @return The index currently pointed at
	 */
	public int i();
}
