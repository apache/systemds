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

import org.apache.sysds.runtime.DMLRuntimeException;

public class H5RuntimeException extends DMLRuntimeException {
	private static final long serialVersionUID = -3551978964353888835L;

	public H5RuntimeException(String string) {
		super(string);
	}

	public H5RuntimeException(Exception e) {
		super(e);
	}

	public H5RuntimeException(String string, Exception ex) {
		super(string, ex);
	}
}
