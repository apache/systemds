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

package org.apache.sysds.cujava.runtime;

public class CudaMemcpyKind {

	/**
	 * Host -> Host
	 */
	public static final int cudaMemcpyHostToHost = 0;

	/**
	 * Host -> Device
	 */
	public static final int cudaMemcpyHostToDevice = 1;

	/**
	 * Device -> Host
	 */
	public static final int cudaMemcpyDeviceToHost = 2;

	/**
	 * Device -> Device
	 */
	public static final int cudaMemcpyDeviceToDevice = 3;

	/**
	 * Autodetect the copy direction (host↔device or device↔device) based on the source and destination pointers.
	 * Requires Unified Virtual Addressing (UVA).
	 */
	public static final int cudaMemcpyDefault = 4;

	private CudaMemcpyKind() {
		// Private constructor to prevent instantiation.
	}
}
