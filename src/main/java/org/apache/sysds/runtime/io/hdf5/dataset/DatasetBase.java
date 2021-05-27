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

import org.apache.sysds.runtime.io.hdf5.AbstractNode;
import org.apache.sysds.runtime.io.hdf5.HdfFileChannel;
import org.apache.sysds.runtime.io.hdf5.ObjectHeader;
import org.apache.sysds.runtime.io.hdf5.api.Dataset;
import org.apache.sysds.runtime.io.hdf5.api.Group;
import org.apache.sysds.runtime.io.hdf5.api.NodeType;
import org.apache.sysds.runtime.io.hdf5.object.datatype.CompoundDataType;
import org.apache.sysds.runtime.io.hdf5.object.datatype.DataType;
import org.apache.sysds.runtime.io.hdf5.object.datatype.OrderedDataType;
import org.apache.sysds.runtime.io.hdf5.object.datatype.VariableLength;
import org.apache.sysds.runtime.io.hdf5.object.message.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static java.nio.ByteOrder.LITTLE_ENDIAN;
import static org.apache.commons.lang3.ClassUtils.primitiveToWrapper;

public abstract class DatasetBase extends AbstractNode implements Dataset {
	private static final Logger logger = LoggerFactory.getLogger(DatasetBase.class);

	protected final HdfFileChannel hdfFc;
	protected final ObjectHeader oh;

	private final DataType dataType;
	private final DataSpace dataSpace;

	public DatasetBase(HdfFileChannel hdfFc, long address, String name, Group parent, ObjectHeader oh) {
		super(hdfFc, address, name, parent);
		this.hdfFc = hdfFc;
		this.oh = oh;

		dataType = getHeaderMessage(DataTypeMessage.class).getDataType();
		dataSpace = getHeaderMessage(DataSpaceMessage.class).getDataSpace();
	}

	@Override
	public NodeType getType() {
		return NodeType.DATASET;
	}

	protected void convertToCorrectEndiness(ByteBuffer bb) {
		if (dataType instanceof OrderedDataType) {
			final ByteOrder order = (((OrderedDataType) dataType).getByteOrder());
			bb.order(order);
			if(logger.isTraceEnabled()) {
				logger.trace("Set buffer order of '{}' to {}", getPath(), order);
			}
		} else {
			bb.order(LITTLE_ENDIAN);
		}
	}

	@Override
	public long getSize() {
		return dataSpace.getTotalLength();
	}

	@Override
	public long getSizeInBytes() {
		return getSize() * dataType.getSize();
	}

	@Override
	public int[] getDimensions() {
		return dataSpace.getDimensions();
	}

	@Override
	public int[] getMaxSize() {
		if (dataSpace.isMaxSizesPresent()) {
			return dataSpace.getMaxSizes();
		} else {
			return getDimensions();
		}
	}

	@Override
	public DataLayout getDataLayout() {
		return getHeaderMessage(DataLayoutMessage.class).getDataLayout();
	}

	@Override
	public Class<?> getJavaType() {
		final Class<?> type = dataType.getJavaType();
		// For scalar datasets the returned type will be the wrapper class because
		// getData returns Object
		if (isScalar() && type.isPrimitive()) {
			return primitiveToWrapper(type);
		}
		return type;
	}

	@Override
	public DataType getDataType() {
		return dataType;
	}

	@Override
	public Object getData() {
		logger.debug("Getting data for '{}'...", getPath());

		if (isEmpty()) {
			return null;
		}

		final ByteBuffer bb = getDataBuffer();
		final DataType type = getDataType();

		return DatasetReader.readDataset(type, bb, getDimensions(), hdfFc);
	}

	@Override
	public boolean isScalar() {
		return getDimensions().length == 0;
	}

	@Override
	public boolean isEmpty() {
		return getSizeInBytes() == 0;
	}

	@Override
	public boolean isCompound() { return getDataType() instanceof CompoundDataType; }

	/**
	 * Gets the buffer that holds this datasets data. The returned buffer will be of
	 * the correct order (endiness).
	 *
	 * @return the data buffer that holds this dataset
	 */
	public abstract ByteBuffer getDataBuffer();

	@Override
	public Object getFillValue() {
		FillValueMessage fillValueMessage = getHeaderMessage(FillValueMessage.class);
		if (fillValueMessage.isFillValueDefined()) {
			ByteBuffer bb = fillValueMessage.getFillValue();
			// Convert to data pass zero length dims for scalar
			return DatasetReader.readDataset(getDataType(), bb, new int[0], hdfFc);
		} else {
			return null;
		}
	}

	@Override
	public String toString() {
		return "DatasetBase [path=" + getPath() + "]";
	}

	@Override
	public boolean isVariableLength() {
		return getDataType() instanceof VariableLength;
	}

	@Override
	public long getStorageInBytes() {
		return getSizeInBytes();
	}
}
