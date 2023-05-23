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

package org.apache.sysds.runtime.compress.lib;

import org.apache.commons.lang.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.AColGroupCompressed;
import org.apache.sysds.runtime.compress.colgroup.ColGroupConst;
import org.apache.sysds.runtime.compress.colgroup.ColGroupDDC;
import org.apache.sysds.runtime.compress.colgroup.ColGroupEmpty;
import org.apache.sysds.runtime.compress.colgroup.dictionary.ADictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.DictionaryFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.estim.encoding.ConstEncoding;
import org.apache.sysds.runtime.compress.estim.encoding.DenseEncoding;
import org.apache.sysds.runtime.compress.estim.encoding.EmptyEncoding;
import org.apache.sysds.runtime.compress.estim.encoding.EncodingFactory;
import org.apache.sysds.runtime.compress.estim.encoding.IEncode;

/**
 * Library functions to combine column groups inside a compressed matrix.
 */
public final class CLALibCombineGroups {
	protected static final Log LOG = LogFactory.getLog(CLALibCombineGroups.class.getName());

	private CLALibCombineGroups() {
		// private constructor
	}

	public static CompressedMatrixBlock combine(CompressedMatrixBlock cmb, int k) {
		throw new NotImplementedException();
	}

	/**
	 * Combine the column groups A and B together.
	 * 
	 * The number of rows should be equal, and it is not verified so there will be unexpected behavior in such cases.
	 * 
	 * @param a The first group to combine.
	 * @param b The second group to combine.
	 * @return A new column group containing the two.
	 */
	public static AColGroup combine(AColGroup a, AColGroup b) {
		IColIndex combinedColumns = ColIndexFactory.combine(a, b);

		if(a instanceof AColGroupCompressed && b instanceof AColGroupCompressed) {
			AColGroupCompressed ac = (AColGroupCompressed) a;
			AColGroupCompressed bc = (AColGroupCompressed) b;
			IEncode ce = EncodingFactory.combine(a, b);

			if(ce instanceof DenseEncoding) {
				DenseEncoding ced = (DenseEncoding) ce;
				ADictionary cd = DictionaryFactory.combineDictionaries(ac, bc);
				return ColGroupDDC.create(combinedColumns, cd, ced.getMap(), null);
			}
			else if(ce instanceof EmptyEncoding) {
				return new ColGroupEmpty(combinedColumns);
			}
			else if(ce instanceof ConstEncoding) {
				ADictionary cd = DictionaryFactory.combineDictionaries(ac, bc);
				return ColGroupConst.create(combinedColumns, cd);
			}
		}

		throw new NotImplementedException(
			"Not implemented combine for " + a.getClass().getSimpleName() + " - " + b.getClass().getSimpleName());

	}

}
