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
import org.apache.sysds.runtime.io.hdf5.api.*;
import org.apache.sysds.runtime.io.hdf5.btree.BTreeV1;
import org.apache.sysds.runtime.io.hdf5.btree.BTreeV2;
import org.apache.sysds.runtime.io.hdf5.btree.record.LinkNameForIndexedGroupRecord;
import org.apache.sysds.runtime.io.hdf5.dataset.DatasetLoader;
import org.apache.sysds.runtime.io.hdf5.exceptions.HdfException;
import org.apache.sysds.runtime.io.hdf5.exceptions.HdfInvalidPathException;
import org.apache.sysds.runtime.io.hdf5.links.ExternalLink;
import org.apache.sysds.runtime.io.hdf5.links.SoftLink;
import org.apache.sysds.runtime.io.hdf5.object.message.*;
import org.apache.commons.lang3.concurrent.ConcurrentException;
import org.apache.commons.lang3.concurrent.LazyInitializer;
import java.nio.ByteBuffer;
import java.util.*;

public class GroupImpl extends AbstractNode implements Group {
	private final class ChildrenLazyInitializer extends LazyInitializer<Map<String, Node>> {
		private final HdfFileChannel hdfFc;
		private final Group parent;

		private ChildrenLazyInitializer(HdfFileChannel hdfFc, Group parent) {
			this.hdfFc = hdfFc;
			this.parent = parent;
		}

		@Override protected Map<String, Node> initialize() throws ConcurrentException {
			if(header.get().hasMessageOfType(SymbolTableMessage.class)) {
				// Its an old style Group
				return createOldStyleGroup(header.get());
			}
			else {
				return createNewStyleGroup(header.get());
			}
		}

		private Map<String, Node> createNewStyleGroup(final ObjectHeader oh) {
			// Need to get a list of LinkMessages
			final List<LinkMessage> links;

			final LinkInfoMessage linkInfoMessage = oh.getMessageOfType(LinkInfoMessage.class);
			if(linkInfoMessage.getBTreeNameIndexAddress() == Constants.UNDEFINED_ADDRESS) {
				// Links stored compactly i.e in the object header, so get directly
				links = oh.getMessagesOfType(LinkMessage.class);
			}
			else {
				// Links are not stored compactly i.e in the fractal heap
				final BTreeV2<LinkNameForIndexedGroupRecord> bTreeNode = new BTreeV2<>(hdfFc,
					linkInfoMessage.getBTreeNameIndexAddress());
				final FractalHeap fractalHeap = new FractalHeap(hdfFc, linkInfoMessage.getFractalHeapAddress());

				List<LinkNameForIndexedGroupRecord> records = bTreeNode.getRecords();
				links = new ArrayList<>(records.size());
				for(LinkNameForIndexedGroupRecord linkName : records) {
					ByteBuffer id = linkName.getId();
					// Get the name data from the fractal heap
					ByteBuffer bb = fractalHeap.getId(id);
					links.add(LinkMessage.fromBuffer(bb, hdfFc.getSuperblock()));
				}
			}

			final Map<String, Node> lazyChildren = new LinkedHashMap<>(links.size());
			for(LinkMessage link : links) {
				String linkName = link.getLinkName();
				switch(link.getLinkType()) {
					case HARD:
						long hardLinkAddress = link.getHardLinkAddress();
						final Node node = createNode(linkName, hardLinkAddress);
						lazyChildren.put(linkName, node);
						break;
					case SOFT:
						lazyChildren.put(linkName, new SoftLink(link.getSoftLink(), linkName, parent));
						break;
					case EXTERNAL:
						lazyChildren.put(linkName,
							new ExternalLink(link.getExternalFile(), link.getExternalPath(), linkName, parent));
						break;
				}
			}

			return lazyChildren;
		}

		private Map<String, Node> createOldStyleGroup(final ObjectHeader oh) {

			Gson gson=new Gson();

			final SymbolTableMessage stm = oh.getMessageOfType(SymbolTableMessage.class);
			final BTreeV1 rootBTreeNode = BTreeV1.createGroupBTree(hdfFc, stm.getbTreeAddress());
			final LocalHeap rootNameHeap = new LocalHeap(hdfFc, stm.getLocalHeapAddress());
			final ByteBuffer nameBuffer = rootNameHeap.getDataBuffer();
			final List<Long> childAddresses = rootBTreeNode.getChildAddresses();
			final Map<String, Node> lazyChildren = new LinkedHashMap<>(childAddresses.size());

			for(long child : childAddresses) {
				GroupSymbolTableNode groupSTE = new GroupSymbolTableNode(hdfFc, child);

				//System.out.println(gson.toJson(rootNameHeap));
				//System.out.println(gson.toJson(groupSTE));
			//	System.out.println("+++++++++++++++++++++++++++++++++++");

				for(SymbolTableEntry ste : groupSTE.getSymbolTableEntries()) {
					String childName = "DS1";//readName(nameBuffer, ste.getLinkNameOffset());
					final Node node;
					switch(ste.getCacheType()) {
						case 0: // No cache
							// Not cached so need to look at header
							final ObjectHeader header;
							try {
								header = ObjectHeader.readObjectHeader(hdfFc, ste.getObjectHeaderAddress());
							}
							catch(HdfException e) {
								// Add context here we know the child name that failed
								throw new HdfException("Failed to read '" + getPath() + childName + "'", e);
							}

							if(header.hasMessageOfType(DataLayoutMessage.class)) {
								node = DatasetLoader.createDataset(hdfFc, header, childName, parent);
							}
							else {
								node = createGroup(hdfFc, ste.getObjectHeaderAddress(), childName, parent);
							}
							break;
						case 1: // Cached group
							node = createGroup(hdfFc, ste.getObjectHeaderAddress(), childName, parent);
							break;
						case 2: // Soft Link
							String target = readName(nameBuffer, ste.getLinkValueOffset());
							node = new SoftLink(target, childName, parent);
							break;
						default:
							throw new HdfException(
								"Unrecognized symbol table entry cache type. Type was: " + ste.getCacheType());
					}
					lazyChildren.put(childName, node);
				}
			}
			return lazyChildren;
		}

		private Node createNode(String name, long address) {
			final ObjectHeader linkHeader = ObjectHeader.readObjectHeader(hdfFc, address);
			final Node node;
			if(linkHeader.hasMessageOfType(DataSpaceMessage.class)) {
				// Its a a Dataset
				node = DatasetLoader.createDataset(hdfFc, linkHeader, name, parent);
			}
			else {
				// Its a group
				node = createGroup(hdfFc, address, name, parent);
			}
			return node;
		}

		private String readName(ByteBuffer bb, int linkNameOffset) {
			bb.position(linkNameOffset);
			return Utils.readUntilNull(bb);
		}
	}

	private final LazyInitializer<Map<String, Node>> children;

	private GroupImpl(HdfFileChannel hdfFc, long address, String name, Group parent) {
		super(hdfFc, address, name, parent);

		children = new ChildrenLazyInitializer(hdfFc, this);
	}

	private GroupImpl(HdfFileChannel hdfFc, long objectHeaderAddress, HDF5File parent) {
		super(hdfFc, objectHeaderAddress, "", parent); // No name special case for root group no name

		// Special case for root group pass parent instead of this
		children = new ChildrenLazyInitializer(hdfFc, parent);
	}

	static Group createGroup(HdfFileChannel hdfFc, long objectHeaderAddress, String name, Group parent) {
		return new GroupImpl(hdfFc, objectHeaderAddress, name, parent);
	}

	static Group createGroupToWrite(byte version, BufferBuilder bb, int numberOfDimensions, int[] dimensions, int[] maxSizes, int sizeOfLengths, int sizeOfOffsets, String datasetName) {
		return  new GroupImpl(version, bb, numberOfDimensions, dimensions, maxSizes, sizeOfLengths, sizeOfOffsets, datasetName);
	}

	public GroupImpl(byte version, BufferBuilder bb, int numberOfDimensions, int[] dimensions, int[] maxSizes, int sizeOfLengths, int sizeOfOffsets, String datasetName){
		super(version, bb, numberOfDimensions, dimensions, maxSizes, sizeOfLengths, sizeOfOffsets, datasetName);
		children=null;
	}

	static Group createRootGroup(HdfFileChannel hdfFc, long objectHeaderAddress, HDF5File file) {
		// Call the special root group constructor
		return new GroupImpl(hdfFc, objectHeaderAddress, file);
	}

	@Override public Map<String, Node> getChildren() {
		try {

			return children.get();
		}
		catch(Exception e) {
			throw new HdfException(
				"Failed to load children for group '" + getPath() + "' at address '" + getAddress() + "'", e);
		}
	}

	@Override public String toString() {
		return "Group [name=" + name + ", path=" + getPath() + ", address=" + Utils.toHex(getAddress()) + "]";
	}

	@Override public String getPath() {
		return super.getPath() + "/";
	}

	@Override public NodeType getType() {
		return NodeType.GROUP;
	}

	@Override public Iterator<Node> iterator() {
		return getChildren().values().iterator();
	}

	@Override public Node getChild(String name) {
		try {
			return children.get().get(name);
		}
		catch(Exception e) {
			throw new HdfException(
				"Failed to load children of group '" + getPath() + "' at address '" + getAddress() + "'", e);
		}
	}

	@Override public Node getByPath(String path) {
		// Try splitting into 2 sections the child of this group and the remaining path
		// to pass down.
		final String[] pathElements = path.split(Constants.PATH_SEPARATOR, 2);
		Node child = getChild(pathElements[0]);
		// If we have a link try to resolve it
		if(child instanceof Link) {
			child = ((Link) child).getTarget();
		}
		if(pathElements.length == 1 && child != null) {
			// There is no remaining path to resolve so we have the result
			return child;
		}
		else if(child instanceof Group) {
			// The next level is also a group so try to keep resolving the remaining path
			return ((Group) child).getByPath(pathElements[1]);
		}
		else {
			// Path can't be resolved
			throw new HdfInvalidPathException(getPath() + path, getFile());
		}

	}

	@Override public Dataset getDatasetByPath(String path) {
		Node node = getByPath(path);
		if(node instanceof Link) {
			node = ((Link) node).getTarget();
		}
		if(node instanceof Dataset) {
			return (Dataset) node;
		}
		else {
			throw new HdfInvalidPathException(getPath() + path, getFile());
		}
	}

	@Override public boolean isLinkCreationOrderTracked() {
		ObjectHeader oh = getHeader();
		if(oh.hasMessageOfType(LinkInfoMessage.class)) {
			// New style, supports link creation tracking but might not be enabled
			return oh.getMessageOfType(LinkInfoMessage.class).isLinkCreationOrderTracked();
		}
		else {
			// Old style no support for link tracking
			return false;
		}
	}

}
