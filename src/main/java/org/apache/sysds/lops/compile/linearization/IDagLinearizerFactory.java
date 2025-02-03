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

package org.apache.sysds.lops.compile.linearization;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import org.apache.sysds.conf.ConfigurationManager;

public class IDagLinearizerFactory {
	public static Log LOG = LogFactory.getLog(IDagLinearizerFactory.class.getName());

	public enum DagLinearizer {
		DEPTH_FIRST, BREADTH_FIRST, MIN_INTERMEDIATE, MAX_PARALLELIZE, AUTO, 
		PIPELINE_DEPTH_FIRST, RESOURCE_AWARE_FAST, RESOURCE_AWARE_OPTIMAL;
	}

	public static IDagLinearizer createDagLinearizer() {
		DagLinearizer type = ConfigurationManager.getLinearizationOrder();
		return createDagLinearizer(type);
	}

	public static IDagLinearizer createDagLinearizer(DagLinearizer type) {
		switch(type) {
			case AUTO:
				return new LinearizerCostBased();
			case BREADTH_FIRST:
				return new LinearizerBreadthFirst();
			case DEPTH_FIRST:
				return new LinearizerDepthFirst();
			case MAX_PARALLELIZE:
				return new LinearizerMaxParallelism();
			case MIN_INTERMEDIATE:
				return new LinearizerMinIntermediates();
			case PIPELINE_DEPTH_FIRST:
				return new LinearizerPipelineAware();
			case RESOURCE_AWARE_FAST:
				return new LinearizerResourceAwareFast();
			case RESOURCE_AWARE_OPTIMAL:
				return new LinearizerResourceAwareOptimal();
			default:
				LOG.warn("Invalid DAG_LINEARIZATION: " + type + ", falling back to DEPTH_FIRST ordering");
				return new LinearizerDepthFirst();
		}
	}
}
