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

import com.google.gson.Gson;
import org.apache.sysds.runtime.io.hdf5.Superblock.SuperblockV0V1;
import org.apache.sysds.runtime.io.hdf5.Superblock.SuperblockV2V3;
import org.apache.sysds.runtime.io.hdf5.api.*;
import org.apache.sysds.runtime.io.hdf5.exceptions.HdfException;
import org.apache.commons.lang3.StringUtils;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.*;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;

public class HDF5File implements Group, AutoCloseable {

	private File file=null;

	private HdfFileChannel hdfFc=null;

	private Group rootGroup=null;

	private Set<HDF5File> openExternalFiles = new HashSet<>();

	public HDF5File() {}

	public HDF5File(Path path) {
		this(path.toFile());
	}

	public HDF5File(URI uri) {
		this(Paths.get(uri).toFile());
	}

	public static HDF5File fromInputStream(InputStream inputStream) {
		try {
			Path tempFile = Files.createTempFile(null, "-stream.hdf5"); // null random file name
			tempFile.toFile().deleteOnExit(); // Auto cleanup
			Files.copy(inputStream, tempFile, StandardCopyOption.REPLACE_EXISTING);
			return new HDF5File(tempFile);
		}
		catch(IOException e) {
			throw new HdfException("Failed to open input stream", e);
		}
	}

	public HDF5File(File hdfFile) {

		this.file = hdfFile;

		try {
			// Sonar would like this closed but we are implementing a file object which
			// needs this channel for operation it is closed when this HdfFile is closed
			FileChannel fc = FileChannel.open(hdfFile.toPath(), StandardOpenOption.READ); // NOSONAR

			// Find out if the file is a HDF5 file
			boolean validSignature = false;
			long offset;
			for(offset = 0; offset < fc.size(); offset = nextOffset(offset)) {
				validSignature = Superblock.verifySignature(fc, offset);
				if(validSignature) {
					break;
				}
			}
			if(!validSignature) {
				throw new HdfException("No valid HDF5 signature found");
			}
			// We have a valid HDF5 file so read the full superblock
			final Superblock superblock = Superblock.readSuperblock(fc, offset);


			// Validate the superblock
			if(superblock.getBaseAddressByte() != offset) {
				throw new HdfException("Invalid superblock base address detected");
			}

			hdfFc = new HdfFileChannel(fc, superblock);

			if(superblock instanceof SuperblockV0V1) {
				SuperblockV0V1 sb = (SuperblockV0V1) superblock;
				SymbolTableEntry ste = new SymbolTableEntry(hdfFc, sb.getRootGroupSymbolTableAddress() - sb.getBaseAddressByte());
				rootGroup = GroupImpl.createRootGroup(hdfFc, ste.getObjectHeaderAddress(), this);
			}
			else if(superblock instanceof SuperblockV2V3) {
				SuperblockV2V3 sb = (SuperblockV2V3) superblock;
				rootGroup = GroupImpl.createRootGroup(hdfFc, sb.getRootGroupObjectHeaderAddress(), this);
			}
		}
		catch(IOException e) {
			throw new HdfException("Failed to open file '" + file.getAbsolutePath() + "' . Is it a HDF5 file?", e);
		}
	}


	private long nextOffset(long offset) {
		if(offset == 0) {
			return 512L;
		}
		return offset * 2;
	}

	public long getUserBlockSize() {
		return hdfFc.getUserBlockSize();
	}

	public ByteBuffer getUserBlockBuffer() {
		return hdfFc.mapNoOffset(0, hdfFc.getUserBlockSize());
	}

	@Override public void close() {
		for(HDF5File externalHDF5File : openExternalFiles) {
			externalHDF5File.close();
		}

		hdfFc.close();
	}

	public long size() {
		return hdfFc.size();
	}

	@Override public boolean isGroup() {
		return true;
	}

	@Override public Map<String, Node> getChildren() {
		return rootGroup.getChildren();
	}

	@Override public String getName() {
		return file.getName();
	}

	@Override public String getPath() {
		return "/";
	}

	@Override public Map<String, Attribute> getAttributes() {
		return rootGroup.getAttributes();
	}

	@Override public Attribute getAttribute(String name) {
		return rootGroup.getAttribute(name);
	}

	@Override public String toString() {
		return "HdfFile [file=" + file.getName() + "]";
	}

	@Override public NodeType getType() {
		return NodeType.FILE;
	}

	@Override public Group getParent() {
		// The file has no parent so return null
		return null;
	}

	@Override public File getFile() {
		return file;
	}

	@Override public long getAddress() {
		return rootGroup.getAddress();
	}

	@Override public Iterator<Node> iterator() {
		return rootGroup.iterator();
	}

	@Override public Node getChild(String name) {
		return rootGroup.getChild(name);
	}

	@Override public Node getByPath(String path) {
		// As its the file its ok to have a leading slash but strip it here to be
		// consistent with other groups
		path = StringUtils.stripStart(path, Constants.PATH_SEPARATOR);
		return rootGroup.getByPath(path);
	}

	@Override public Dataset getDatasetByPath(String path) {
		// As its the file its ok to have a leading slash but strip it here to be
		// consistent with other groups
		path = StringUtils.stripStart(path, Constants.PATH_SEPARATOR);
		return rootGroup.getDatasetByPath(path);
	}

	@Override public HDF5File getHdfFile() {
		return this;
	}

	public void addExternalFile(HDF5File HDF5File) {
		openExternalFiles.add(HDF5File);
	}

	@Override public boolean isLink() {
		return false;
	}

	@Override public boolean isAttributeCreationOrderTracked() {
		return rootGroup.isAttributeCreationOrderTracked();
	}

	@Override public boolean isLinkCreationOrderTracked() {
		return rootGroup.isLinkCreationOrderTracked();
	}

	public HdfFileChannel getHdfChannel() {
		return hdfFc;
	}

}
