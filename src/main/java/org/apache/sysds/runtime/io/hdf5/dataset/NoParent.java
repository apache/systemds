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


package org.apache.sysds.runtime.io.hdf5.dataset;

import org.apache.sysds.runtime.io.hdf5.HDF5File;
import org.apache.sysds.runtime.io.hdf5.api.*;

import java.io.File;
import java.util.Iterator;
import java.util.Map;


public enum NoParent implements Group {

	INSTANCE;

	private static final String UNKNOWN_GROUP = "Unknown group";
	private static final String UNKNOWN_NAME = "Unknown";

	@Override
	public Map<String, Node> getChildren() {
		throw new UnsupportedOperationException(UNKNOWN_GROUP);
	}

	@Override
	public Node getChild(String name) {
		throw new UnsupportedOperationException(UNKNOWN_GROUP);
	}

	@Override
	public Node getByPath(String path) {
		throw new UnsupportedOperationException(UNKNOWN_GROUP);
	}

	@Override
	public Dataset getDatasetByPath(String path) {
		throw new UnsupportedOperationException(UNKNOWN_GROUP);
	}

	@Override
	public boolean isLinkCreationOrderTracked() {
		throw new UnsupportedOperationException(UNKNOWN_GROUP);
	}

	@Override
	public Group getParent() {
		return INSTANCE;
	}

	@Override
	public String getName() {
		return UNKNOWN_NAME;
	}

	@Override
	public String getPath() {
		return UNKNOWN_NAME + "/";
	}

	@Override
	public Map<String, Attribute> getAttributes() {
		throw new UnsupportedOperationException(UNKNOWN_GROUP);
	}

	@Override
	public Attribute getAttribute(String name) {
		throw new UnsupportedOperationException(UNKNOWN_GROUP);
	}

	@Override
	public NodeType getType() {
		throw new UnsupportedOperationException(UNKNOWN_GROUP);
	}

	@Override
	public boolean isGroup() {
		return true;
	}

	@Override
	public File getFile() {
		throw new UnsupportedOperationException(UNKNOWN_GROUP);
	}

	@Override
	public HDF5File getHdfFile() {
		throw new UnsupportedOperationException(UNKNOWN_GROUP);
	}

	@Override
	public long getAddress() {
		throw new UnsupportedOperationException(UNKNOWN_GROUP);
	}

	@Override
	public boolean isLink() {
		throw new UnsupportedOperationException(UNKNOWN_GROUP);
	}

	@Override
	public boolean isAttributeCreationOrderTracked() {
		throw new UnsupportedOperationException(UNKNOWN_GROUP);
	}

	@Override
	public Iterator<Node> iterator() {
		throw new UnsupportedOperationException(UNKNOWN_GROUP);
	}
}
