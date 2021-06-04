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
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.BitSet;
import java.util.List;

import org.apache.sysds.runtime.io.hdf5.message.H5SymbolTableMessage;

public class H5 {

	// H5 format write/read steps:
	// 1. Create/Open a File (H5Fcreate)
	// 2. Create/open a Dataspace
	// 3. Create/Open a Dataset
	// 4. Write/Read
	// 5. Close File

	// Create a file
	public static H5RootObject H5Fcreate(String fileName) {

		H5RootObject rootObject = new H5RootObject();
		try {
			File file = new File(fileName);
			Files.deleteIfExists(file.toPath());
			file.createNewFile();
			FileChannel fc = FileChannel.open(file.toPath(), StandardOpenOption.WRITE);
			rootObject.setFileChannel(fc);
			rootObject.bufferBuilder=new H5BufferBuilder();
		}
		catch(Exception exception) {
			throw new H5Exception("Can't create fine in path " + fileName);
		}
		return rootObject;
	}

	// Open a file and data space
	public static H5RootObject H5Fopen(Path path) {
		File file = path.toFile();
		return H5.H5Fopen(file);
	}

	public static H5RootObject H5Fopen(String fileName) {
		try {
			File file = new File(fileName);
			return H5.H5Fopen(file);
		}
		catch(Exception exception){
			throw new H5Exception(exception);
		}
	}

	public static H5RootObject H5Fopen(File file) {

		H5RootObject rootObject = new H5RootObject();
		try {
			FileChannel fc = FileChannel.open(file.toPath(), StandardOpenOption.READ);

			// Find out if the file is a HDF5 file
			boolean validSignature = false;
			long offset;
			for(offset = 0; offset < fc.size(); offset = nextOffset(offset)) {
				validSignature = H5Superblock.verifySignature(fc, offset);
				if(validSignature) {
					break;
				}
			}
			if(!validSignature) {
				throw new H5Exception("No valid HDF5 signature found");
			}
			rootObject.setFileChannel(fc);

			final H5Superblock superblock = new H5Superblock(rootObject.getFileChannel(), offset);
			rootObject.setSuperblock(superblock);
		}
		catch(Exception exception) {
			exception.printStackTrace();
			//throw new H5Exception("Can't open fine in path " + fileName);
		}
		return rootObject;
	}

	private static long nextOffset(long offset) {
		if(offset == 0) {
			return 512L;
		}
		return offset * 2;
	}

	// Create Data Space
	public static void H5Screate(H5RootObject rootObject, long row, long col) {

		try {
			final H5Superblock superblock = new H5Superblock();
			superblock.versionOfSuperblock = 0;
			superblock.versionNumberOfTheFileFreeSpaceInformation = 0;
			superblock.versionOfRootGroupSymbolTableEntry = 0;
			superblock.versionOfSharedHeaderMessageFormat = 0;
			superblock.sizeOfOffsets = 8;
			superblock.sizeOfLengths = 8;
			superblock.groupLeafNodeK = 4;
			superblock.groupInternalNodeK = 16;
			superblock.baseAddressByte = 0;
			superblock.addressOfGlobalFreeSpaceIndex = -1;
			superblock.endOfFileAddress = 2048 + (row * col * 8); // double value
			superblock.driverInformationBlockAddress = -1;
			superblock.rootGroupSymbolTableAddress = 56;

			rootObject.setSuperblock(superblock);
			rootObject.setRank(2);
			rootObject.setCol(col);
			rootObject.setRow(row);

			superblock.toBuffer(rootObject.bufferBuilder);

			H5SymbolTableEntry symbolTableEntry = new H5SymbolTableEntry(rootObject);
			symbolTableEntry.toBuffer(rootObject.bufferBuilder);

			rootObject.getFileChannel().write(rootObject.bufferBuilder.build());
		}
		catch(Exception exception) {
			throw new H5Exception(exception);
		}
	}

	// Open a Data Space
	public static H5ContiguousDataset H5Dopen(H5RootObject rootObject, String datasetName) {
		try {
			H5SymbolTableEntry symbolTableEntry = new H5SymbolTableEntry(rootObject,
				rootObject.getSuperblock().rootGroupSymbolTableAddress - rootObject.getSuperblock().baseAddressByte);

			H5ObjectHeader objectHeader = new H5ObjectHeader(rootObject, symbolTableEntry.getObjectHeaderAddress());

			final H5SymbolTableMessage stm = (H5SymbolTableMessage) objectHeader.getMessages().get(0);
			final H5BTree rootBTreeNode = new H5BTree(rootObject, stm.getbTreeAddress());
			final H5LocalHeap rootNameHeap = new H5LocalHeap(rootObject, stm.getLocalHeapAddress());
			final ByteBuffer nameBuffer = rootNameHeap.getDataBuffer();
			final List<Long> childAddresses = rootBTreeNode.getChildAddresses();

			long child = childAddresses.get(0);

			H5GroupSymbolTableNode groupSTE = new H5GroupSymbolTableNode(rootObject, child);

			symbolTableEntry = groupSTE.getSymbolTableEntries()[0];

			nameBuffer.position(symbolTableEntry.getLinkNameOffset());
			String childName = Utils.readUntilNull(nameBuffer);

			if(!childName.equals(datasetName)) {
				throw new H5Exception("The dataset name '" + datasetName + "' not found!");
			}

			final H5ObjectHeader header = new H5ObjectHeader(rootObject, symbolTableEntry.getObjectHeaderAddress());
			final H5ContiguousDataset contiguousDataset = new H5ContiguousDataset(rootObject, header);
			return contiguousDataset;

		}
		catch(Exception exception) {
			throw new H5Exception(exception);
		}
	}

	// Create Dataset
	public static void H5Dcreate(H5RootObject rootObject, long maxRow, long maxCol, String datasetName) {

		if(rootObject.getRank() == 2) {

			rootObject.setMaxRow(maxRow);
			rootObject.setMaxCol(maxCol);
			rootObject.setDatasetName(datasetName);
			H5ObjectHeader objectHeader = new H5ObjectHeader(rootObject, datasetName);
			objectHeader.toBuffer(rootObject.bufferBuilder);
			try {
				rootObject.bufferBuilder.goToPositionWithWriteZero(2048);
				rootObject.fileChannel.position(0);
				rootObject.getFileChannel().write(rootObject.bufferBuilder.build());
			}
			catch(Exception exception) {
				throw new H5Exception(exception);
			}
		}
		else
			throw new H5Exception("Just support Matrix!");
	}

	// Write Data
	public static void H5Dwrite(H5RootObject rootObject, double[] data) {

	}
	public static void H5Dwrite(H5RootObject rootObject, double[][] data) {

		long dataSize = rootObject.getRow() * rootObject.getCol() * 8;
		int maxBufferSize = 100000;//Integer.MAX_VALUE;
		int bufferSize = (int) Math.min(maxBufferSize, dataSize);

		H5BufferBuilder bb = new H5BufferBuilder();
		try {
			for(int i = 0; i < rootObject.getRow(); i++) {
				for(int j = 0; j < rootObject.getCol(); j++) {
//					if(bb.getSize() + 8 > bufferSize) {
//						rootObject.getFileChannel().write(bb.build());
//						bb = new H5BufferBuilder();
//					}
					bb.writeDouble(data[i][j]);
				}
			}

//			BitSet bitSet1=new BitSet(64);
//			BitSet bitSet2=new BitSet(64);
//			BitSet bitSet3=new BitSet(64);
//			BitSet bitSet4=new BitSet(64);
//
//
//			//2
//			bitSet1.set(1);
//			bb.writeBitSet(bitSet1,64);
//			//3
//			bitSet2.set(1);
//			bb.writeBitSet(bitSet2,64);
//			//4
//			bitSet3.set(1);
//			bb.writeBitSet(bitSet3,64);
//			//5
//			bitSet4.set(1);
//			bb.writeBitSet(bitSet4,64);


			rootObject.getFileChannel().write(bb.build());
		}
		catch(Exception exception) {
			throw new H5Exception(exception);
		}
	}
	public static void H5Fclose(H5RootObject rootObject){
		try {
			rootObject.getFileChannel().close();
		}
		catch(Exception exception){
			throw  new H5Exception(exception);
		}
	}

	public static double[][] H5Dread(H5RootObject rootObject, H5ContiguousDataset dataset) {

		ByteBuffer buffer = dataset.getDataBuffer();
		int[] dimensions = rootObject.getDimensions();
		final double[][] data = dataset.getDataType().getDoubleDataType().fillData(buffer, dimensions);
		return data;
	}

}
