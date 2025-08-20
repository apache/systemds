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

package org.apache.sysds.cujava.interop;

import java.lang.reflect.Field;

public class JCudaAdapter {
	private JCudaAdapter() {}

	public static jcuda.Pointer toJCuda(org.apache.sysds.cujava.Pointer p) {
		try {
			jcuda.Pointer q = new jcuda.Pointer();

			// jcuda.NativePointerObject.nativePointer = cuJava nativePointer
			Field np = jcuda.NativePointerObject.class.getDeclaredField("nativePointer");
			np.setAccessible(true);
			np.setLong(q, p.getNativePointer());

			// jcuda.Pointer.byteOffset = cuJava byteOffset
			Field bo = jcuda.Pointer.class.getDeclaredField("byteOffset");
			bo.setAccessible(true);
			bo.setLong(q, p.getByteOffset());

			return q;
		} catch (ReflectiveOperationException e) {
			throw new IllegalStateException("cuJava→JCuda pointer adaptation failed", e);
		}
	}
}
