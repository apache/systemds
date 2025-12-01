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

package org.apache.sysds.api;

import java.util.List;
import java.util.Set;

import org.apache.sysds.api.mlcontext.ScriptExecutor;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.hops.codegen.SpoofCompiler;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.Program;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.instructions.gpu.context.GPUContext;
import org.apache.sysds.runtime.instructions.gpu.context.GPUContextPool;
import org.apache.sysds.runtime.instructions.gpu.context.GPUObject;
import org.apache.sysds.runtime.lineage.LineageEstimatorStatistics;
import org.apache.sysds.runtime.lineage.LineageGPUCacheEviction;
import org.apache.sysds.runtime.ooc.cache.OOCCacheManager;
import org.apache.sysds.utils.Statistics;

public class ScriptExecutorUtils {

	/**
	 * Execute the runtime program. This involves execution of the program
	 * blocks that make up the runtime program and may involve dynamic
	 * recompilation.
	 * 
	 * @param se
	 *            script executor
	 * @param statisticsMaxHeavyHitters
	 *            maximum number of statistics to print
	 */
	public static void executeRuntimeProgram(ScriptExecutor se, int statisticsMaxHeavyHitters) {
		Program prog = se.getRuntimeProgram();
		ExecutionContext ec = se.getExecutionContext();
		DMLConfig config = se.getConfig();
		executeRuntimeProgram(prog, ec, config, statisticsMaxHeavyHitters, se.getScript().getOutputVariables());
	}

	/**
	 * Execute the runtime program. This involves execution of the program
	 * blocks that make up the runtime program and may involve dynamic
	 * recompilation.
	 * 
	 * @param rtprog
	 *            runtime program
	 * @param ec
	 *            execution context
	 * @param dmlconf
	 *            dml configuration
	 * @param statisticsMaxHeavyHitters
	 *            maximum number of statistics to print
	 * @param outputVariables
	 *            output variables that were registered as part of MLContext
	 */
	public static void executeRuntimeProgram(Program rtprog, ExecutionContext ec, DMLConfig dmlconf, int statisticsMaxHeavyHitters, Set<String> outputVariables) {
		Statistics.startRunTimer();
		try {
			// run execute (w/ exception handling to ensure proper shutdown)
			if (DMLScript.USE_ACCELERATOR && ec != null) {
				List<GPUContext> gCtxs = GPUContextPool.reserveAllGPUContexts();
				if (gCtxs == null) {
					throw new DMLRuntimeException(
							"GPU : Could not create GPUContext, either no GPU or all GPUs currently in use");
				}
				gCtxs.get(0).initializeThread();
				ec.setGPUContexts(gCtxs);
			}
			rtprog.execute(ec);
		} catch (Throwable e) {
			throw e;
		} finally { // ensure cleanup/shutdown
			if (DMLScript.USE_ACCELERATOR && !ec.getGPUContexts().isEmpty()) {
				// -----------------------------------------------------------------
				// The below code pulls the output variables on the GPU to the host. This is required especially when:
				// The output variable was generated as part of a MLContext session with GPU enabled
				// and was passed to another MLContext with GPU disabled
				// The above scenario occurs in our gpu test suite (eg: BatchNormTest).
				if(outputVariables != null) {
					for(String outVar : outputVariables) {
						Data data = ec.getVariable(outVar);
						if(data != null && data instanceof MatrixObject) {
							for(GPUContext gCtx : ec.getGPUContexts()) {
								GPUObject gpuObj = ((MatrixObject)data).getGPUObject(gCtx);
								if(gpuObj != null && gpuObj.isDirty()) {
									gpuObj.acquireHostRead(null);
								}
							}
						}
					}
				}
				// -----------------------------------------------------------------
				for(GPUContext gCtx : ec.getGPUContexts()) {
					gCtx.clearTemporaryMemory();
				}
				GPUContextPool.freeAllGPUContexts();
				if (LineageGPUCacheEviction.gpuEvictionThread != null)
					LineageGPUCacheEviction.gpuEvictionThread.shutdown();
			}
			if( ConfigurationManager.isCodegenEnabled() )
				SpoofCompiler.cleanupCodeGenerator();
			
			// display statistics (incl caching stats if enabled)
			Statistics.stopRunTimer();
			System.out.println(Statistics.display(statisticsMaxHeavyHitters > 0 ?
					statisticsMaxHeavyHitters : DMLScript.STATISTICS_COUNT));
			
			if (DMLScript.LINEAGE_ESTIMATE)
				System.out.println(LineageEstimatorStatistics.displayLineageEstimates());

			if (DMLScript.USE_OOC)
				OOCCacheManager.reset();
		}
	}

}
