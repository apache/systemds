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


package org.apache.sysds.runtime.io.hdf5.object.message;

import org.apache.sysds.runtime.io.hdf5.BufferBuilder;
import org.apache.sysds.runtime.io.hdf5.exceptions.HdfException;
import org.apache.sysds.runtime.io.hdf5.exceptions.UnsupportedHdfException;
import org.apache.sysds.runtime.io.hdf5.object.datatype.*;

import java.nio.ByteBuffer;
import java.util.BitSet;

public class DataTypeMessage extends Message {

	private DataType dataType;

	public DataTypeMessage(ByteBuffer bb, BitSet flags) {
		super(flags);

		dataType = DataType.readDataType(bb);
	}

	public DataTypeMessage(int dataClass, BitSet flags) {
		super(flags);
		switch(dataClass) {
			case 0: // Fixed point
				//dataType = FixedPoint(bb);
				break;
			case 1: // Floating point
				dataType = new FloatingPoint();
				break;
			case 2: // Time
				break;
			case 3: // String
				//dataType = StringData(bb);
				break;
			case 4: // Bit field
				//dataType = BitField(bb);
				break;
			case 5: // Opaque
				throw new UnsupportedHdfException("Opaque data type is not yet supported");
			case 6: // Compound
				//dataType =CompoundDataType(bb);
				break;
			case 7: // Reference
				//dataType = Reference(bb);
				break;
			case 8: // Enum
				//dataType = EnumDataType(bb);
				break;
			case 9: // Variable length
				//dataType = VariableLength(bb);
				break;
			case 10: // Array
				//dataType = ArrayDataType(bb);
				break;
			default:
				throw new HdfException("Unrecognized data class = " + dataClass);
		}
	}


	public DataType getDataType() {
		return dataType;
	}

	public BufferBuilder toBuffer() {
		BufferBuilder header = new BufferBuilder();
		return toBuffer(header);
	}

	public BufferBuilder toBuffer(BufferBuilder header) {
	 	dataType.toBuffer(header);
		return header;
	}

}
