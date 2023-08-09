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

package org.apache.sysds.runtime.compress.colgroup.scheme;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.Pair;

/**
 * Interface for a scheme instance.
 * 
 * Instances of this class has the purpose of encoding the minimum values required to reproduce a compression scheme,
 * and apply it to unseen data.
 * 
 * The reproduced compression scheme should be able to apply to unseen data and multiple instance of data into the same
 * compression plan, and in extension make it possible to continuously extend already compressed data representations.
 * 
 * A single scheme is only responsible for encoding a single column group type.
 */
public interface ICLAScheme {

	/** Logging access for the CLA Scheme encoders */
	public final Log LOG = LogFactory.getLog(ICLAScheme.class.getName());

	/**
	 * Encode the given matrix block into the scheme provided in the instance.
	 * 
	 * 
	 * The method is unsafe in the sense that if the encoding scheme does not fit, there is no guarantee that an error
	 * is thrown. To guarantee the encoding scheme, first use update on the matrix block and used the returned scheme to
	 * ensure consistency.
	 * 
	 * @param data The data to encode
	 * @throws IllegalArgumentException In the case the columns argument number of columns does not corelate with the
	 *                                  schemes list of columns.
	 * @return A compressed column group forced to use the scheme provided.
	 */
	public AColGroup encode(MatrixBlock data);

	/**
	 * Encode a given matrix block into the scheme provided in the instance but overwrite what columns to use.
	 * 
	 * The method is unsafe in the sense that if the encoding scheme does not fit, there is no guarantee that an error
	 * is thrown. To guarantee the encoding scheme, first use update on the matrix block and used the returned scheme to
	 * ensure consistency.
	 * 
	 * @param data    The data to encode
	 * @param columns The columns to apply the scheme to, but must be of same number than the encoded scheme
	 * @throws IllegalArgumentException In the case the columns argument number of columns does not corelate with the
	 *                                  schemes list of columns.
	 * @return A compressed column group forced to use the scheme provided.
	 */
	public AColGroup encode(MatrixBlock data, IColIndex columns);

	/**
	 * Update the encoding scheme to enable compression of the given data.
	 * 
	 * @param data The data to update into the scheme
	 * @return A updated scheme
	 */
	public ICLAScheme update(MatrixBlock data);

	/**
	 * Update the encoding scheme to enable compression of the given data.
	 * 
	 * @param data    The data to update into the scheme
	 * @param columns The columns to extract the data from
	 * @return A updated scheme
	 */
	public ICLAScheme update(MatrixBlock data, IColIndex columns);

	/**
	 * Update and encode the given block in a single pass. It can fail to do so in cases where the dictionary size
	 * increase over the mapping sizes supported by individual encodings.
	 * 
	 * The implementation should always work and fall back to a normal two pass algorithm if it breaks.
	 * 
	 * @param data The block to encode
	 * @return The updated scheme and an encoded columngroup
	 */
	public Pair<ICLAScheme, AColGroup> updateAndEncode(MatrixBlock data);

	/**
	 * Try to update and encode in a single pass over the data. It can fail to do so in cases where the dictionary size
	 * increase over the mapping sizes supported by individual encodings.
	 * 
	 * The implementation should always work and fall back to a normal two pass algorithm if it breaks.
	 * 
	 * @param data    The block to encode
	 * @param columns The column to encode
	 * @return The updated scheme and an encoded columngroup
	 */
	public Pair<ICLAScheme, AColGroup> updateAndEncode(MatrixBlock data, IColIndex columns);
}
