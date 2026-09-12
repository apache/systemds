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

package org.apache.sysds.cujava;

public class Sizeof {

	/**
	 * CUDA expects sizes in bytes. The JDK provides sizes in bits.
	 * Hence, we divide the sizes provided by the JDK by 8 to obtain bytes.
	 */

	public static final int BYTE = Byte.SIZE / 8;

	public static final int CHAR = Character.SIZE / 8;

	public static final int SHORT = Short.SIZE / 8;

	public static final int INT = Integer.SIZE / 8;

	public static final int FLOAT = Float.SIZE / 8;

	public static final int LONG = Long.SIZE / 8;

	public static final int DOUBLE = Double.SIZE / 8;

	// Keep constructor private to prevent instantiation
	private Sizeof() {
	}

}
