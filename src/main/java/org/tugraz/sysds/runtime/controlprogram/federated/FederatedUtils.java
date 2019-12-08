/*
 * Copyright 2019 Graz University of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package org.tugraz.sysds.runtime.controlprogram.federated;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.tugraz.sysds.conf.ConfigurationManager;
import org.tugraz.sysds.hops.OptimizerUtils;
import org.tugraz.sysds.runtime.controlprogram.caching.TensorObject;
import org.tugraz.sysds.runtime.data.TensorBlock;
import org.tugraz.sysds.runtime.meta.MetaData;
import org.tugraz.sysds.runtime.meta.TensorCharacteristics;

public class FederatedUtils {
	protected static final Log LOG = LogFactory.getLog(FederatedUtils.class.getName());
	public static final String FEDERATED_FUNC_PREFIX = "_fed_";
	
	public static TensorObject newTensorObject(TensorBlock tb, boolean cleanup) {
		TensorObject result = new TensorObject(OptimizerUtils.getUniqueTempFileName(),
				new MetaData(new TensorCharacteristics(new long[]{-1, -1}, ConfigurationManager.getBlocksize())));
		result.acquireModify(tb);
		result.release();
		result.enableCleanup(cleanup);
		return result;
	}
}
