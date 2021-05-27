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

package org.apache.sysds.runtime.io.hdf5;

public class HDF5Constants {

	public static final int H5F_ACC_TRUNC;
	public static final String VERSION_V0_V1;
	public static final String VERSION_V2_V3;
	public static final long H5I_INVALID_HID;

	static {
		H5F_ACC_TRUNC = -1;
		VERSION_V0_V1 = "v0v1";
		VERSION_V2_V3 = "v2v3";
		H5I_INVALID_HID = -1;
	}

}
