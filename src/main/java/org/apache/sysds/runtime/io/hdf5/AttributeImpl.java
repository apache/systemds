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
import org.apache.sysds.runtime.io.hdf5.api.Node;
import org.apache.sysds.runtime.io.hdf5.dataset.DatasetReader;
import org.apache.sysds.runtime.io.hdf5.object.datatype.DataType;
import org.apache.sysds.runtime.io.hdf5.object.message.AttributeMessage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.ByteBuffer;

import static org.apache.commons.lang3.ClassUtils.primitiveToWrapper;

public class AttributeImpl implements Attribute {
	private static final Logger logger = LoggerFactory.getLogger(AttributeImpl.class);

	private final HdfFileChannel hdfFc;
	private final Node node;
	private final String name;
	private final AttributeMessage message;

	public AttributeImpl(HdfFileChannel hdfFc, Node node, AttributeMessage message) {
		this.hdfFc = hdfFc;
		this.node = node;
		this.name = message.getName();
		this.message = message;
	}

	@Override
	public Node getNode() {
		return node;
	}

	@Override
	public String getName() {
		return this.name;
	}

	@Override
	public long getSize() {
		return message.getDataSpace().getTotalLength();
	}

	@Override
	public long getSizeInBytes() {
		return getSize() * message.getDataType().getSize();
	}

	@Override
	public int[] getDimensions() {
		return message.getDataSpace().getDimensions();
	}

	@Override
	public Object getData() {
		logger.debug("Getting data for attribute '{}' of '{}'...", name, node.getPath());
		if (isEmpty()) {
			return null;
		}
		DataType type = message.getDataType();
		ByteBuffer bb = message.getDataBuffer();
		return DatasetReader.readDataset(type, bb, getDimensions(), hdfFc);
	}

	@Override
	public boolean isEmpty() {
		return message.getDataBuffer() == null;
	}

	@Override
	public boolean isScalar() {
		if (isEmpty()) {
			return false;
		}
		return getDimensions().length == 0;
	}

	@Override
	public Class<?> getJavaType() {
		final Class<?> type = message.getDataType().getJavaType();
		// For scalar datasets the returned type will be the wrapper class because
		// getData returns Object
		if (isScalar() && type.isPrimitive()) {
			return primitiveToWrapper(type);
		}
		return type;
	}

	@Override
	public ByteBuffer getBuffer() {
		return message.getDataBuffer();
	}
}
