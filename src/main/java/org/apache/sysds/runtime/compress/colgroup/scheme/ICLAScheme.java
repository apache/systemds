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
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

/**
 * Abstract class for a scheme instance.
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
	 * The method returns null, if it is impossible to encode the input data with the given scheme.
	 * 
	 * @param data The data to encode
	 * @return A compressed column group or null.
	 */
	public AColGroup encode(MatrixBlock data);

	/**
	 * Encode a given matrix block into the scheme provided in the instance but overwrite what columns to use.
	 * 
	 * The method returns null, if it is impossible to encode the input data with the given scheme.
	 * 
	 * @param data    The data to encode
	 * @param columns The columns to apply the scheme to, but must be of same number than the encoded scheme
	 * @throws IllegalArgumentException In the case the columns argument number of columns doesent corelate with the
	 *                                  Schemes list of columns.
	 * @return A compressed column group or null
	 */
	public AColGroup encode(MatrixBlock data, int[] columns);
}
