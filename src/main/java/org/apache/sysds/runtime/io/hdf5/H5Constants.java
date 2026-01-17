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

public final class H5Constants {
	public static final byte NULL = '\0';
	public static final long UNDEFINED_ADDRESS = -1L;
	public static final int STATIC_HEADER_SIZE = 2048;
	public static final int NIL_MESSAGE = 0;
	public static final int DATA_SPACE_MESSAGE = 1;
	public static final int DATA_TYPE_MESSAGE = 3;
	public static final int FILL_VALUE_MESSAGE = 5;
	public static final int DATA_LAYOUT_MESSAGE = 8;
	public static final int SYMBOL_TABLE_MESSAGE = 17;
	public static final int OBJECT_MODIFICATION_TIME_MESSAGE = 18;
	public static final int FILTER_PIPELINE_MESSAGE = 11;
	public static final int ATTRIBUTE_MESSAGE = 12;
}
