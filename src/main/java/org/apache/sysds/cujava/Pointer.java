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

import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.nio.ShortBuffer;
import java.nio.IntBuffer;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.nio.DoubleBuffer;
import java.nio.ByteOrder;

public class Pointer extends NativePointerObject {

	private long byteOffset;
	private final Buffer buffer;
	private final NativePointerObject[] pointers;

	public Pointer() {
		buffer = null;
		pointers = null;
		byteOffset = 0;
	}

	protected Pointer(long nativePointerValue) {
		super(nativePointerValue);
		buffer = null;
		pointers = null;
		byteOffset = 0;
	}

	private Pointer(Buffer buffer) {
		this.buffer = buffer;
		pointers = null;
		byteOffset = 0;
	}

	private Pointer(NativePointerObject[] pointers) {
		buffer = null;
		this.pointers = pointers;
		byteOffset = 0;
	}

	protected Pointer(Pointer other) {
		super(other.getNativePointer());
		this.buffer = other.buffer;
		this.pointers = other.pointers;
		this.byteOffset = other.byteOffset;
	}

	protected Pointer(Pointer other, long byteOffset) {
		this(other);
		this.byteOffset += byteOffset;
	}

	public static Pointer to(byte[] values) {
		return new Pointer(ByteBuffer.wrap(values));
	}

	public static Pointer to(char[] values) {
		return new Pointer(CharBuffer.wrap(values));
	}

	public static Pointer to(short[] values) {
		return new Pointer(ShortBuffer.wrap(values));
	}

	public static Pointer to(int[] values) {
		return new Pointer(IntBuffer.wrap(values));
	}

	public static Pointer to(float[] values) {
		return new Pointer(FloatBuffer.wrap(values));
	}

	public static Pointer to(long[] values) {
		return new Pointer(LongBuffer.wrap(values));
	}

	public static Pointer to(double[] values) {
		return new Pointer(DoubleBuffer.wrap(values));
	}

	public static Pointer to(NativePointerObject... pointers) {
		if(pointers == null) {
			throw new IllegalArgumentException(
				"The pointers argument is null – expected one or more NativePointerObject references.");
		}
		return new Pointer(pointers);
	}

	public Pointer withByteOffset(long byteOffset) {
		return new Pointer(this, byteOffset);
	}

	public long getByteOffset() {
		return byteOffset;
	}

	public long address() {                      // nativePointer + byteOffset
		return getNativePointer() + getByteOffset();
	}

	public ByteBuffer getByteBuffer(long byteOffset, long byteSize) {
		if(buffer == null) {
			return null;
		}
		if(!(buffer instanceof ByteBuffer internalByteBuffer)) {
			return null;
		}
		ByteBuffer byteBuffer = internalByteBuffer.slice();
		byteBuffer.limit(Math.toIntExact(byteOffset + byteSize));
		byteBuffer.position(Math.toIntExact(byteOffset));
		return byteBuffer.slice().order(ByteOrder.nativeOrder());
	}

	public static Pointer to(Buffer buffer) {
		if(buffer == null || (!buffer.isDirect() && !buffer.hasArray())) {
			throw new IllegalArgumentException(
				"Invalid buffer: argument is null or neither direct nor backed by an array; " +
					"expected a non-null direct buffer or one with an accessible backing array.");
		}
		return new Pointer(buffer);
	}

}
