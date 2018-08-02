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
import java.util.Set;

import org.apache.sysml.api.mlcontext.ScriptExecutor;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.conf.DMLConfig;
import org.apache.sysml.hops.codegen.SpoofCompiler;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.Program;
import org.apache.sysml.runtime.controlprogram.caching.CacheableData;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysml.runtime.instructions.cp.Data;
import org.apache.sysml.runtime.instructions.gpu.context.GPUContext;
import org.apache.sysml.runtime.instructions.gpu.context.GPUContextPool;
import org.apache.sysml.runtime.instructions.gpu.context.GPUObject;
import org.apache.sysml.utils.NativeHelper;
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
		// Whether extra statistics useful for developers and others interested
		// in digging into performance problems are recorded and displayed
		DMLScript.FINEGRAINED_STATISTICS = DMLScript.STATISTICS && dmlconf.getBooleanValue(DMLConfig.EXTRA_FINEGRAINED_STATS);
		DMLScript.PRINT_GPU_MEMORY_INFO = dmlconf.getBooleanValue(DMLConfig.PRINT_GPU_MEMORY_INFO);
		DMLScript.SYNCHRONIZE_GPU = dmlconf.getBooleanValue(DMLConfig.SYNCHRONIZE_GPU);
		CacheableData.CACHING_BUFFER_SIZE = dmlconf.getDoubleValue(DMLConfig.CACHING_BUFFER_SIZE);
		if(CacheableData.CACHING_BUFFER_SIZE < 0 || CacheableData.CACHING_BUFFER_SIZE > 1) 
			throw new RuntimeException("Incorrect value (" + CacheableData.CACHING_BUFFER_SIZE + ") for the configuration " + DMLConfig.CACHING_BUFFER_SIZE);
		DMLScript.EAGER_CUDA_FREE = dmlconf.getBooleanValue(DMLConfig.EAGER_CUDA_FREE);
		DMLScript.STATISTICS_MAX_WRAP_LEN = dmlconf.getIntValue(DMLConfig.STATS_MAX_WRAP_LEN);		
		NativeHelper.initialize(dmlconf.getTextValue(DMLConfig.NATIVE_BLAS_DIR), dmlconf.getTextValue(DMLConfig.NATIVE_BLAS).trim());
		
		if(DMLScript.USE_ACCELERATOR) {
			DMLScript.FLOATING_POINT_PRECISION = dmlconf.getTextValue(DMLConfig.FLOATING_POINT_PRECISION);
			org.apache.sysml.runtime.matrix.data.LibMatrixCUDA.resetFloatingPointPrecision();
			if(DMLScript.FLOATING_POINT_PRECISION.equals("double")) {
				DMLScript.EVICTION_SHADOW_BUFFER_MAX_BYTES = 0;
			}
			else {
				double shadowBufferSize = dmlconf.getDoubleValue(DMLConfig.EVICTION_SHADOW_BUFFERSIZE);
				if(shadowBufferSize < 0 || shadowBufferSize > 1) 
					throw new RuntimeException("Incorrect value (" + shadowBufferSize + ") for the configuration " + DMLConfig.EVICTION_SHADOW_BUFFERSIZE);
				DMLScript.EVICTION_SHADOW_BUFFER_MAX_BYTES = (long) (((double)InfrastructureAnalyzer.getLocalMaxMemory())*shadowBufferSize);
				if(DMLScript.EVICTION_SHADOW_BUFFER_MAX_BYTES > 0 && 
						DMLScript.EVICTION_SHADOW_BUFFER_CURR_BYTES > DMLScript.EVICTION_SHADOW_BUFFER_MAX_BYTES) {
					// This will be printed in a very rare situation when:
					// 1. There is a memory leak which leads to non-cleared shadow buffer OR
					// 2. MLContext is registering to bunch of outputs that are all part of shadow buffer
					System.out.println("WARN: Cannot use the shadow buffer due to potentially cached GPU objects. Current shadow buffer size (in bytes):" 
						+ DMLScript.EVICTION_SHADOW_BUFFER_CURR_BYTES + " > Max shadow buffer size (in bytes):" + DMLScript.EVICTION_SHADOW_BUFFER_MAX_BYTES);
				}
			}
		}
		

		boolean exceptionThrown = false;

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
			exceptionThrown = true;
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
			}
			if( ConfigurationManager.isCodegenEnabled() )
				SpoofCompiler.cleanupCodeGenerator();
			
			// display statistics (incl caching stats if enabled)
			Statistics.stopRunTimer();
			(exceptionThrown ? System.err : System.out)
				.println(Statistics.display(statisticsMaxHeavyHitters > 0 ?
					statisticsMaxHeavyHitters : DMLScript.STATISTICS_COUNT));
		}
	}

}
