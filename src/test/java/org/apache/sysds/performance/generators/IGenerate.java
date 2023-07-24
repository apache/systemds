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

package org.apache.sysds.performance.generators;

/**
 * Generator interface for task generation.
 */
public interface IGenerate<T> {

	/**
	 * Validate if the generator is empty, and we have to wait for elements.
	 * 
	 * @return If the generator is empty
	 */
	public boolean isEmpty();

	/**
	 * Default wait time for the generator to fill
	 * 
	 * @return The wait time
	 */
	public int defaultWaitTime();

	/**
	 * A Blocking take operation that waits for the Generator to fill that element
	 * 
	 * @return An task element
	 */
	public T take();

	/**
	 * A Non blocking async operation that generates elements for the task que
	 * 
	 * @param N The number of elements to create
	 * @throws InterruptedException An exception if the task is interrupted
	 */
	public void generate(int N) throws InterruptedException;

}
