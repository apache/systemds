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
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.BitSet;
import static java.nio.ByteOrder.LITTLE_ENDIAN;

public class H5BufferBuilder {

	private final ByteArrayOutputStream byteArrayOutputStream;
	private final DataOutputStream dataOutputStream;
	private ByteOrder byteOrder = LITTLE_ENDIAN;

	public H5BufferBuilder() {
		this.byteArrayOutputStream = new ByteArrayOutputStream();
		this.dataOutputStream = new DataOutputStream(byteArrayOutputStream);
	}

	public int getSize(){
		return dataOutputStream.size();
	}

	public void writeByte(int i) {
		try {
			dataOutputStream.writeByte(i);
		}
		catch(IOException e) {
			throw new H5Exception(e);
		}
	}

	public void writeBytes(byte[] bytes) {
		try {
			dataOutputStream.write(bytes);
		}
		catch(IOException e) {
			throw new H5Exception(e);
		}
	}

	public void writeInt(int i) {
		try {
			if(byteOrder == LITTLE_ENDIAN) {
				i = Integer.reverseBytes(i);
			}
			dataOutputStream.writeInt(i);
		}
		catch(IOException e) {
			throw new H5Exception(e);
		}
	}

	public void writeShort(short i) {
		try {
			if(byteOrder == LITTLE_ENDIAN) {
				i = Short.reverseBytes(i);
			}
			dataOutputStream.writeShort(i);
		}
		catch(IOException e) {
			throw new H5Exception(e);
		}
	}

	public void writeLong(long l) {
		try {
			if(byteOrder == LITTLE_ENDIAN) {
				l = Long.reverseBytes(l);
			}
			dataOutputStream.writeLong(l);
		}
		catch(IOException e) {
			throw new H5Exception(e);
		}
	}

	public BitSet convert(long value) {
		BitSet bits = new BitSet();
		int index = 0;
		while (value != 0L) {
			if (value % 2L != 0) {
				bits.set(index);
			}
			++index;
			value = value >>> 1;
		}
		return bits;
	}

	public void writeDouble(double l) {
		try {
			//System.out.println(byteOrder.toString());
			//this.byteOrder=BIG_ENDIAN;
//			if(byteOrder == LITTLE_ENDIAN) {
//				Double s=10d;
				//l = Float.r
//				Long.reverseBytes()
//				l = Double.re r.reverseBytes(l);
//			}
			//l = Double.doubleToLongBits(l);
			//int intBits = Float.floatToIntBits(value);
			//        String binary = Integer.toBinaryString(intBits);
			//        return binary;

//			long longBits = Double.doubleToLongBits(l);
//			//BitSet bs = BitSet.valueOf();
//			//String s= Long.toBinaryString(longBits);
//			BitSet bs= convert(longBits);
//			this.writeBytes(bs.toByteArray());
//			//this.writeBitSet(bs,bs.length());
//			System.out.println(l);
//			System.out.println(bs.length());

			//l = Double.doubleToRawLongBits(l);
			//System.out.println(l);
			//long dtl = Double.dou(l);//Double.doubleToLongBits(l);
			//Double.parseDouble()

			//dtl = Long.reverseBytes(dtl);


			//double d = Double.longBitsToDouble(dtl);
			//dataOutputStream.writeDouble(dtl);
			//System.out.println("value="+l+"  iee="+dtl);
			//this.writeLong((long) dtl);
			//dataOutputStream.writeDouble(l);

			byte[] b = getFloat64(l);
			this.writeBytes(b);
		}
		catch(Exception e) {
			throw new H5Exception(e);
		}
	}

	public static byte[] getFloat64(double value)
	{
		final byte[] float64Bytes = new byte[8];
		long double64Long=Double.doubleToLongBits(value);
		float64Bytes[0] = (byte)((double64Long >> 56) & 0xff);
		float64Bytes[1] = (byte)((double64Long >> 48) & 0xff);
		float64Bytes[2] = (byte)((double64Long >> 40) & 0xff);
		float64Bytes[3] = (byte)((double64Long >> 32) & 0xff);
		float64Bytes[4] = (byte)((double64Long >> 24) & 0xff);
		float64Bytes[5] = (byte)((double64Long >> 16) & 0xff);
		float64Bytes[6] = (byte)((double64Long >> 8) & 0xff);
		float64Bytes[7] = (byte)((double64Long >> 0) & 0xff);
		return float64Bytes;
	}

	public void write(long v, int sizeOfLength){
		try {
			switch(sizeOfLength){
				case 2:
					this.writeShort((short) v);
					break;
				case 4:
					this.writeInt((int) v);
					break;
				case 8:
					this.writeLong(v);
					break;

			}
		}
		catch(Exception e){
			throw new H5Exception(e);
		}

	}

	public ByteBuffer build() {
		try {
			ByteBuffer byteBuffer = ByteBuffer.wrap(byteArrayOutputStream.toByteArray());
			byteBuffer.order(byteOrder);
			dataOutputStream.close();
			byteArrayOutputStream.close();
			return byteBuffer;
		}
		catch(IOException e) {
			throw new H5Exception(e);
		}
	}

	public void writeBitSet(BitSet bitSet, int length) {
		if(bitSet.length() > length) {
			throw new H5Exception("BitSet is longer than length provided");
		}
		try {
			final byte[] bytes = Arrays.copyOf(bitSet.toByteArray(), length); // Ensure empty Bitset are not shortened
			dataOutputStream.write(bytes);
		}
		catch(IOException e) {
			throw new H5Exception(e);
		}
	}

	public void goToPositionWithWriteZero(long pos) {
		long gap = pos - this.dataOutputStream.size();
		byte[] gapByte = new byte[(int) gap];
		this.writeBytes(gapByte);
	}

}
