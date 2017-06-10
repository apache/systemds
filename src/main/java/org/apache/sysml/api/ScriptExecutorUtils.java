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

package org.apache.sysml.api;

import java.util.List;

import org.apache.sysml.api.mlcontext.ScriptExecutor;
import org.apache.sysml.conf.DMLConfig;
import org.apache.sysml.hops.codegen.SpoofCompiler;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.Program;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.instructions.gpu.context.GPUContext;
import org.apache.sysml.runtime.instructions.gpu.context.GPUContextPool;
import org.apache.sysml.runtime.matrix.data.LibMatrixDNN;
import org.apache.sysml.utils.GPUStatistics;
import org.apache.sysml.utils.Statistics;

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
	 * @throws DMLRuntimeException
	 *             if exception occurs
	 */
	public static void executeRuntimeProgram(ScriptExecutor se, int statisticsMaxHeavyHitters) throws DMLRuntimeException {
		Program prog = se.getRuntimeProgram();
		ExecutionContext ec = se.getExecutionContext();
		DMLConfig config = se.getConfig();
		executeRuntimeProgram(prog, ec, config, statisticsMaxHeavyHitters);
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
	 * @throws DMLRuntimeException
	 *             if error occurs
	 */
	public static void executeRuntimeProgram(Program rtprog, ExecutionContext ec, DMLConfig dmlconf, int statisticsMaxHeavyHitters)
			throws DMLRuntimeException {
		// Whether extra statistics useful for developers and others interested
		// in digging into performance problems are recorded and displayed
		GPUStatistics.DISPLAY_STATISTICS = dmlconf.getBooleanValue(DMLConfig.EXTRA_GPU_STATS);
		LibMatrixDNN.DISPLAY_STATISTICS = dmlconf.getBooleanValue(DMLConfig.EXTRA_DNN_STATS);

		// Sets the maximum number of GPUs per process, -1 for all available
		// GPUs
		GPUContextPool.PER_PROCESS_MAX_GPUS = dmlconf.getIntValue(DMLConfig.MAX_GPUS_PER_PROCESS);
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
		} finally { // ensure cleanup/shutdown
			if (DMLScript.USE_ACCELERATOR && !ec.getGPUContexts().isEmpty()) {
				ec.getGPUContexts().forEach(gCtx -> gCtx.clearTemporaryMemory());
				GPUContextPool.freeAllGPUContexts();
			}
			if (dmlconf.getBooleanValue(DMLConfig.CODEGEN))
				SpoofCompiler.cleanupCodeGenerator();

			// display statistics (incl caching stats if enabled)
			Statistics.stopRunTimer();

			if(statisticsMaxHeavyHitters > 0)
				System.out.println(Statistics.display(statisticsMaxHeavyHitters));
			else
				System.out.println(Statistics.display());
		}
	}

}
