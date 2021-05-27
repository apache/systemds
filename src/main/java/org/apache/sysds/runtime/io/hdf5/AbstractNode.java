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

import org.apache.sysds.runtime.io.hdf5.api.Attribute;
import org.apache.sysds.runtime.io.hdf5.api.Group;
import org.apache.sysds.runtime.io.hdf5.api.Node;
import org.apache.sysds.runtime.io.hdf5.api.NodeType;
import org.apache.sysds.runtime.io.hdf5.btree.BTreeV2;
import org.apache.sysds.runtime.io.hdf5.btree.record.AttributeNameForIndexedAttributesRecord;
import org.apache.sysds.runtime.io.hdf5.exceptions.HdfException;
import org.apache.sysds.runtime.io.hdf5.object.message.AttributeInfoMessage;
import org.apache.sysds.runtime.io.hdf5.object.message.AttributeMessage;
import org.apache.sysds.runtime.io.hdf5.object.message.Message;
import org.apache.commons.lang3.concurrent.ConcurrentException;
import org.apache.commons.lang3.concurrent.LazyInitializer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static java.util.stream.Collectors.toMap;

public abstract class AbstractNode implements Node {
	private static final Logger logger = LoggerFactory.getLogger(AbstractNode.class);

	protected final class AttributesLazyInitializer extends LazyInitializer<Map<String, Attribute>> {
		private final LazyInitializer<ObjectHeader> lazyObjectHeader;

		public AttributesLazyInitializer(LazyInitializer<ObjectHeader> lazyObjectHeader) {
			this.lazyObjectHeader = lazyObjectHeader;
		}

		@Override protected Map<String, Attribute> initialize() throws ConcurrentException {
			logger.debug("Lazy initializing attributes for '{}'", getPath());
			final ObjectHeader oh = lazyObjectHeader.get();

			List<AttributeMessage> attributeMessages = new ArrayList<>();

			if(oh.hasMessageOfType(AttributeInfoMessage.class)) {
				// Attributes stored in b-tree
				AttributeInfoMessage attributeInfoMessage = oh.getMessageOfType(AttributeInfoMessage.class);

				if(attributeInfoMessage.getFractalHeapAddress() != Constants.UNDEFINED_ADDRESS) {
					// Create the heap and btree
					FractalHeap fractalHeap = new FractalHeap(hdfFc, attributeInfoMessage.getFractalHeapAddress());
					BTreeV2<AttributeNameForIndexedAttributesRecord> btree = new BTreeV2<>(hdfFc,
						attributeInfoMessage.getAttributeNameBTreeAddress());

					// Read the attribute messages from the btree+heap
					for(AttributeNameForIndexedAttributesRecord attributeRecord : btree.getRecords()) {
						ByteBuffer bb = fractalHeap.getId(attributeRecord.getHeapId());
						AttributeMessage attributeMessage = new AttributeMessage(bb, hdfFc.getSuperblock(),
							attributeRecord.getFlags());
						logger.trace("Read attribute message '{}'", attributeMessage);
						attributeMessages.add(attributeMessage);
					}
				}
			}

			// Add the messages stored directly in the header
			attributeMessages.addAll(oh.getMessagesOfType(AttributeMessage.class));

			return attributeMessages.stream().collect(
				toMap(AttributeMessage::getName, message -> new AttributeImpl(hdfFc, AbstractNode.this, message)));
		}
	}

	private  HdfFileChannel hdfFc;
	protected  long address;
	protected  String name;
	protected  Group parent;
	protected  LazyInitializer<ObjectHeader> header;
	protected  AttributesLazyInitializer attributes;

	public AbstractNode(HdfFileChannel hdfFc, long address, String name, Group parent) {
		this.hdfFc = hdfFc;
		this.address = address;
		this.name = name;
		this.parent = parent;

		try {
			header = ObjectHeader.lazyReadObjectHeader(hdfFc, address);

			// Attributes
			attributes = new AttributesLazyInitializer(header);
		}
		catch(Exception e) {
			throw new HdfException("Error reading node '" + getPath() + "' at address " + address, e);
		}
	}

	// for write
	public AbstractNode(byte version, BufferBuilder bb, int numberOfDimensions, int[] dimensions, int[] maxSizes, int sizeOfLengths, int sizeOfOffsets, String childName){
		ObjectHeader.writeObjectHeader(version, bb, numberOfDimensions, dimensions, maxSizes, sizeOfLengths, sizeOfOffsets, childName);
	}

	@Override public boolean isGroup() {
		return getType() == NodeType.GROUP;
	}

	@Override public String getName() {
		return name;
	}

	@Override public String getPath() {
		return parent.getPath() + name;
	}

	@Override public Group getParent() {
		return parent;
	}

	@Override public long getAddress() {
		return address;
	}

	@Override public File getFile() {
		// Recurse back up to the file
		return getParent().getFile();
	}

	@Override public HDF5File getHdfFile() {
		return getParent().getHdfFile();
	}

	@Override public boolean isLink() {
		return false;
	}

	protected <T extends Message> T getHeaderMessage(Class<T> clazz) {
		return getHeader().getMessageOfType(clazz);
	}

	@Override public Map<String, Attribute> getAttributes() {
		try {
			return attributes.get();
		}
		catch(Exception e) {
			throw new HdfException(
				"Failed to load attributes for '" + getPath() + "' at address '" + getAddress() + "'", e);
		}
	}

	@Override public Attribute getAttribute(String name) {
		return getAttributes().get(name);
	}

	@Override public boolean isAttributeCreationOrderTracked() {
		return getHeader().isAttributeCreationOrderTracked();
	}

	public ObjectHeader getHeader() {
		try {
			return header.get();
		}
		catch(Exception e) {
			throw new HdfException("Failed reading header for '" + getPath() + "' at address '" + getAddress() + "'",
				e);
		}
	}
}
