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

package org.apache.sysds.runtime.compress.colgroup;

/**
 * Interface for frame of reference groups.
 */
public interface IFrameOfReferenceGroup extends IContainDefaultTuple {
	/**
	 * extract common value from group and return non morphing group
	 * 
	 * @param constV a vector to contain all values, note length = nCols in total matrix.
	 * @return A non morphing column group with decompression instructions.
	 */
	public AColGroup extractCommon(double[] constV);

}
