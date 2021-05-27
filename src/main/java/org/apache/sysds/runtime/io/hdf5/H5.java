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

import java.io.File;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.StandardOpenOption;

import com.google.gson.Gson;
import org.apache.sysds.runtime.io.hdf5.Superblock.SuperblockV0V1;
import org.apache.sysds.runtime.io.hdf5.Superblock.SuperblockV2V3;

public class H5 {

	public static long H5Fcreate(String fileName, byte version, int numberOfDimensions, long row, long col, long maxRow,
		long maxCol, String datasetName) {

		try {
			File file = new File(fileName);
			file.createNewFile();

			FileChannel fc = FileChannel.open(file.toPath(), StandardOpenOption.WRITE);
			BufferBuilder bufferBuilder;

			// Level 0 - File Metadata
			//1. Disk Format: Level 0A - Format Signature and Superblock
			final Superblock superblock = Superblock.createSuperblock(version, row, col);

			if(superblock instanceof Superblock.SuperblockV0V1) {
				SuperblockV0V1 sb = (SuperblockV0V1) superblock;
				bufferBuilder = sb.toBuffer();
			}
			else if(superblock instanceof Superblock.SuperblockV2V3) {
				SuperblockV2V3 sb = (SuperblockV2V3) superblock;
				bufferBuilder = sb.toBuffer();
			}
			else
				return HDF5Constants.H5I_INVALID_HID;

			// Add Symbol Table Entry
			HdfFileChannel hdfFc = new HdfFileChannel(fc, superblock);
			SymbolTableEntry symbolTableEntry = new SymbolTableEntry(hdfFc);
			symbolTableEntry.toBuffer(bufferBuilder);

			// Add Header Messages

			if(numberOfDimensions == 2) {
				int[] dimensions = {(int) row, (int) col};
				int[] maxSizes = null;
				if(maxCol != 0 && maxRow != 0 && maxRow != -1 && maxCol != -1) {
					maxSizes = new int[numberOfDimensions];
					maxSizes[0] = (int) maxRow;
					maxSizes[1] = (int) maxCol;
				}
				GroupImpl.createGroupToWrite(version, bufferBuilder, numberOfDimensions, dimensions, maxSizes,
					superblock.getSizeOfLengths(), superblock.getSizeOfOffsets(), datasetName);
			}

			fc.write(bufferBuilder.build());
			fc.close();
		}
		catch(Exception ex) {
			ex.printStackTrace();
		}
		return HDF5Constants.H5I_INVALID_HID;
	}
}
