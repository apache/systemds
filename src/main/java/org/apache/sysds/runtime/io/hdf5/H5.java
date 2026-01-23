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

import java.io.BufferedOutputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import org.apache.sysds.runtime.io.hdf5.message.H5SymbolTableMessage;

public class H5 {

	// H5 format write/read steps:
	// 1. Create/Open a File (H5Fcreate)
	// 2. Create/open a Dataspace
	// 3. Create/Open a Dataset
	// 4. Write/Read
	// 5. Close File

	public static H5RootObject H5Fopen(H5ByteReader reader) {
		H5RootObject rootObject = new H5RootObject();
		try {
			// Find out if the file is a HDF5 file
			int maxSignatureLength = 2048;
			boolean validSignature = false;
			long offset;
			for(offset = 0; offset < maxSignatureLength; offset = nextOffset(offset)) {
				validSignature = H5Superblock.verifySignature(reader, offset);
				if(validSignature) {
					break;
				}
			}
			if(!validSignature) {
				throw new H5RuntimeException("No valid HDF5 signature found");
			}
			rootObject.setByteReader(reader);

			final H5Superblock superblock = new H5Superblock(reader, offset);
			rootObject.setSuperblock(superblock);
		}
		catch(Exception exception) {
			throw new H5RuntimeException("Can't open fine " + exception);
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
	public static H5RootObject H5Screate(BufferedOutputStream bos, long row, long col) {

		try {
			H5RootObject rootObject = new H5RootObject();
			rootObject.setBufferedOutputStream(bos);
			rootObject.bufferBuilder = new H5BufferBuilder();
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

			return rootObject;

		}
		catch(Exception exception) {
			throw new H5RuntimeException(exception);
		}
	}

	// Open a Data Space
	public static H5ContiguousDataset H5Dopen(H5RootObject rootObject, String datasetName) {
		try {
			List<String> datasetPath = normalizeDatasetPath(datasetName);
			H5SymbolTableEntry currentEntry = new H5SymbolTableEntry(rootObject,
				rootObject.getSuperblock().rootGroupSymbolTableAddress - rootObject.getSuperblock().baseAddressByte);
			rootObject.setDatasetName(datasetName);

			StringBuilder traversedPath = new StringBuilder("/");
			for(String segment : datasetPath) {
				currentEntry = descendIntoChild(rootObject, currentEntry, segment, traversedPath.toString());
				if(traversedPath.length() > 1)
					traversedPath.append('/');
				traversedPath.append(segment);
			}

			if(H5RootObject.HDF5_DEBUG) {
				System.out.println("[HDF5] Opening dataset '" + datasetName + "' resolved to object header @ "
					+ currentEntry.getObjectHeaderAddress());
			}

			final H5ObjectHeader header = new H5ObjectHeader(rootObject, currentEntry.getObjectHeaderAddress());
			return new H5ContiguousDataset(rootObject, header);

		}
		catch(Exception exception) {
			throw new H5RuntimeException(exception);
		}
	}

	private static H5SymbolTableEntry descendIntoChild(H5RootObject rootObject, H5SymbolTableEntry parentEntry,
		String childSegment, String currentPath) {
		H5ObjectHeader objectHeader = new H5ObjectHeader(rootObject, parentEntry.getObjectHeaderAddress());
		H5SymbolTableMessage symbolTableMessage = objectHeader.getMessageOfType(H5SymbolTableMessage.class);
		List<H5SymbolTableEntry> children = readSymbolTableEntries(rootObject, symbolTableMessage);
		H5LocalHeap heap = new H5LocalHeap(rootObject, symbolTableMessage.getLocalHeapAddress());
		ByteBuffer nameBuffer = heap.getDataBuffer();
		List<String> availableNames = new ArrayList<>();
		for(H5SymbolTableEntry child : children) {
			nameBuffer.position(child.getLinkNameOffset());
			String candidateName = Utils.readUntilNull(nameBuffer);
			if(H5RootObject.HDF5_DEBUG) {
				System.out.println("[HDF5] Visit '" + currentPath + "' child -> '" + candidateName + "'");
			}
			availableNames.add(candidateName);
			if(candidateName.equals(childSegment)) {
				return child;
			}
		}
		throw new H5RuntimeException("Dataset path segment '" + childSegment + "' not found under '" + currentPath
			+ "'. Available entries: " + availableNames);
	}

	private static List<H5SymbolTableEntry> readSymbolTableEntries(H5RootObject rootObject,
		H5SymbolTableMessage symbolTableMessage) {
		H5BTree btree = new H5BTree(rootObject, symbolTableMessage.getbTreeAddress());
		List<H5SymbolTableEntry> entries = new ArrayList<>();
		for(Long childAddress : btree.getChildAddresses()) {
			H5GroupSymbolTableNode groupNode = new H5GroupSymbolTableNode(rootObject, childAddress);
			entries.addAll(Arrays.asList(groupNode.getSymbolTableEntries()));
		}
		return entries;
	}

	private static List<String> normalizeDatasetPath(String datasetName) {
		if(datasetName == null) {
			throw new H5RuntimeException("Dataset name cannot be null");
		}
		List<String> tokens = Arrays.stream(datasetName.split("/"))
			.map(String::trim)
			.filter(token -> !token.isEmpty())
			.collect(Collectors.toList());
		if(tokens.isEmpty()) {
			throw new H5RuntimeException("Dataset name '" + datasetName + "' is invalid.");
		}
		return tokens;
	}

	// Create Dataset
	public static void H5Dcreate(H5RootObject rootObject, long maxRow, long maxCol, String datasetName) {

		if(rootObject.getRank() == 2) {

			rootObject.setMaxRow(maxRow);
			rootObject.setMaxCol(maxCol);
			rootObject.setDatasetName(datasetName);
			H5ObjectHeader objectHeader = new H5ObjectHeader(rootObject, datasetName);
			objectHeader.toBuffer(rootObject.bufferBuilder);
			rootObject.bufferBuilder.goToPositionWithWriteZero(2048);

		}
		else
			throw new H5RuntimeException("Just support Matrix!");
	}

	public static void H5WriteHeaders(H5RootObject rootObject) {
		try {
			rootObject.getBufferedOutputStream().write(rootObject.bufferBuilder.build().array());
		}
		catch(Exception exception) {
			throw new H5RuntimeException(exception);
		}
	}

	// Write Data
	public static void H5Dwrite(H5RootObject rootObject, double[] data) {
		try {
			H5BufferBuilder bb = new H5BufferBuilder();
			for(Double d : data) {
				bb.writeDouble(d);
			}
			rootObject.getBufferedOutputStream().write(bb.noOrderBuild().array());
		}
		catch(Exception exception) {
			throw new H5RuntimeException(exception);
		}
	}

	public static void H5Dwrite(H5RootObject rootObject, double[][] data) {

		for(int i = 0; i < rootObject.getRow(); i++) {
			H5Dwrite(rootObject, data[i]);
		}
	}

	public static void H5Dread(H5RootObject rootObject, H5ContiguousDataset dataset, double[][] data) {
		for(int i = 0; i < rootObject.getRow(); i++) {
			dataset.readRowDoubles(i, data[i], 0);
		}
	}

	public static void H5Dread(H5ContiguousDataset dataset, int row, double[] data) {
		dataset.readRowDoubles(row, data, 0);
	}

}
