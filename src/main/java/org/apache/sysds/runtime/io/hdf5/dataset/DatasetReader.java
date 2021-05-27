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

import org.apache.sysds.runtime.io.hdf5.HdfFileChannel;
import org.apache.sysds.runtime.io.hdf5.object.datatype.DataType;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;

/**
 * <p>
 * This class handles converting the {@link ByteBuffer} obtained from the file
 * into a Java array containing the data. It makes use of Java NIO ByteBuffers
 * bulk read methods where possible to enable high performance IO.
 * </p>
 * Some useful information about HDF5 → Java type mappings see:
 * <ul>
 * <li><a href=
 * "https://support.hdfgroup.org/ftp/HDF5/prev-releases/HDF-JAVA/hdfjni-3.2.1/hdf5_java_doc/hdf/hdf5lib/H5.html">HDF5
 * Java wrapper H5.java</a></li>
 * <li><a href="http://docs.h5py.org/en/stable/faq.html">h5py FAQ</a></li>
 * <li><a href=
 * "https://docs.oracle.com/javase/tutorial/java/nutsandbolts/datatypes.html">Java
 * primitive types</a></li>
 * </ul>
 *
 * @author James Mudd
 */
public final class DatasetReader {

	private DatasetReader() {
		throw new AssertionError("No instances of DatasetReader");
	}

	/**
	 * This converts a buffer into a Java object representing this dataset.
	 *
	 * @param type The data type of this dataset
	 * @param buffer The buffer containing the dataset
	 * @param dimensions The dimensions of this dataset
	 * @param hdfFc The file channel for reading the file
	 * @return A Java object representation of this dataset
	 */
	public static Object readDataset(DataType type, ByteBuffer buffer, int[] dimensions, HdfFileChannel hdfFc) {
		// If the data is scalar make a fake one element array then remove it at the end

		final boolean isScalar;
		if (dimensions.length == 0) {
			// Scalar dataset
			isScalar = true;
			dimensions = new int[]{1}; // Fake the dimensions
		} else {
			isScalar = false;
		}

		final Object data = type.fillData(buffer, dimensions, hdfFc);

		if (isScalar) {
			return Array.get(data, 0);
		} else {
			return data;
		}
	}
}
