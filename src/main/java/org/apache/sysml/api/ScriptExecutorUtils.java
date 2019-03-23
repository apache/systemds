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

import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

import org.apache.sysml.api.jmlc.JMLCUtils;
import org.apache.sysml.api.mlcontext.MLContextUtil;
import org.apache.sysml.api.mlcontext.ScriptType;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.conf.DMLConfig;
import org.apache.sysml.hops.codegen.SpoofCompiler;
import org.apache.sysml.hops.rewrite.ProgramRewriter;
import org.apache.sysml.hops.rewrite.RewriteRemovePersistentReadWrite;
import org.apache.sysml.parser.DMLProgram;
import org.apache.sysml.parser.DMLTranslator;
import org.apache.sysml.parser.LanguageException;
import org.apache.sysml.parser.ParseException;
import org.apache.sysml.parser.ParserFactory;
import org.apache.sysml.parser.ParserWrapper;
import org.apache.sysml.runtime.controlprogram.LocalVariableMap;
import org.apache.sysml.runtime.controlprogram.Program;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContextFactory;
import org.apache.sysml.runtime.instructions.cp.Data;
import org.apache.sysml.runtime.instructions.gpu.context.GPUContext;
import org.apache.sysml.runtime.instructions.gpu.context.GPUObject;
import org.apache.sysml.runtime.util.UtilFunctions;
import org.apache.sysml.utils.Explain;
import org.apache.sysml.utils.Statistics;
import org.apache.sysml.utils.Explain.ExplainCounts;
import org.apache.sysml.utils.Explain.ExplainType;
import org.apache.sysml.yarn.DMLAppMasterUtils;
import org.apache.sysml.yarn.DMLYarnClientProxy;
import org.apache.sysml.runtime.DMLRuntimeException;

public class ScriptExecutorUtils {

	public static final boolean IS_JCUDA_AVAILABLE;
	static {
		// Early detection of JCuda libraries avoids synchronization overhead for common JMLC scenario:
		// i.e. CPU-only multi-threaded execution
		boolean isJCudaAvailable = false;
		try {
			Class.forName("jcuda.Pointer");
			isJCudaAvailable = true;
		}
		catch (ClassNotFoundException e) { }
		IS_JCUDA_AVAILABLE = isJCudaAvailable;
	}

	public static enum SystemMLAPI {
		DMLScript,
		MLContext,
		JMLC
	}

	public static Program compileRuntimeProgram(String script, Map<String,String> nsscripts, Map<String, String> args,
												String[] inputs, String[] outputs, ScriptType scriptType, DMLConfig dmlconf, SystemMLAPI api) {
		return compileRuntimeProgram(script, nsscripts, args, null, null, inputs, outputs,
				scriptType, dmlconf, api, true, false, false);
	}

	public static Program compileRuntimeProgram(String script, Map<String, String> args, String[] allArgs,
												ScriptType scriptType, DMLConfig dmlconf, SystemMLAPI api) {
		return compileRuntimeProgram(script, Collections.emptyMap(), args, allArgs, null, null, null,
				scriptType, dmlconf, api, true, false, false);
	}

	/**
	 * Compile a runtime program
	 *
	 * @param script string representing of the DML or PyDML script
	 * @param nsscripts map (name, script) of the DML or PyDML namespace scripts
	 * @param args map of input parameters ($) and their values
	 * @param allArgs commandline arguments
	 * @param symbolTable symbol table associated with MLContext
	 * @param inputs string array of input variables to register
	 * @param outputs string array of output variables to register
	 * @param scriptType is this script DML or PyDML
	 * @param dmlconf configuration provided by the user
	 * @param api API used to execute the runtime program
	 * @param performHOPRewrites should perform hop rewrites
	 * @param maintainSymbolTable whether or not all values should be maintained in the symbol table after execution.
	 * @param init whether to initialize hadoop execution
	 * @return compiled runtime program
	 */
	public static Program compileRuntimeProgram(String script, Map<String,String> nsscripts, Map<String, String> args, String[] allArgs,
												// Input/Outputs registered in MLContext and JMLC. These are set to null by DMLScript
												LocalVariableMap symbolTable, String[] inputs, String[] outputs,
												ScriptType scriptType, DMLConfig dmlconf, SystemMLAPI api,
												// MLContext-specific flags
												boolean performHOPRewrites, boolean maintainSymbolTable,
												boolean init) {
		DMLScript.SCRIPT_TYPE = scriptType;

		Program rtprog;

		if (ConfigurationManager.isGPU() && !IS_JCUDA_AVAILABLE)
			throw new RuntimeException("Incorrect usage: Cannot use the GPU backend without JCuda libraries. Hint: Include systemml-*-extra.jar (compiled using mvn package -P distribution) into the classpath.");
		else if (!ConfigurationManager.isGPU() && ConfigurationManager.isForcedGPU())
			throw new RuntimeException("Incorrect usage: Cannot force a GPU-execution without enabling GPU");

		if(api == SystemMLAPI.JMLC) {
			//check for valid names of passed arguments
			String[] invalidArgs = args.keySet().stream()
					.filter(k -> k==null || !k.startsWith("$")).toArray(String[]::new);
			if( invalidArgs.length > 0 )
				throw new LanguageException("Invalid argument names: "+Arrays.toString(invalidArgs));

			//check for valid names of input and output variables
			String[] invalidVars = UtilFunctions.asSet(inputs, outputs).stream()
					.filter(k -> k==null || k.startsWith("$")).toArray(String[]::new);
			if( invalidVars.length > 0 )
				throw new LanguageException("Invalid variable names: "+Arrays.toString(invalidVars));
		}

		String dmlParserFilePath = (api == SystemMLAPI.JMLC) ? null : DMLScript.DML_FILE_PATH_ANTLR_PARSER;

		try {
			//Step 1: set local/remote memory if requested (for compile in AM context)
			if(api == SystemMLAPI.DMLScript && dmlconf.getBooleanValue(DMLConfig.YARN_APPMASTER) ){
				DMLAppMasterUtils.setupConfigRemoteMaxMemory(dmlconf);
			}

			// Start timer (disabled for JMLC)
			if(api != SystemMLAPI.JMLC)
				Statistics.startCompileTimer();

			//Step 2: parse dml script
			ParserWrapper parser = ParserFactory.createParser(scriptType, nsscripts);
			DMLProgram prog = parser.parse(dmlParserFilePath, script, args);

			//Step 3: construct HOP DAGs (incl LVA, validate, and setup)
			DMLTranslator dmlt = new DMLTranslator(prog);
			dmlt.liveVariableAnalysis(prog);
			dmlt.validateParseTree(prog);
			dmlt.constructHops(prog);

			//init working directories (before usage by following compilation steps)
			if(api != SystemMLAPI.JMLC)
				if (api != SystemMLAPI.MLContext || init)
					DMLScript.initHadoopExecution( dmlconf );


			//Step 4: rewrite HOP DAGs (incl IPA and memory estimates)
			if(performHOPRewrites)
				dmlt.rewriteHopsDAG(prog);

			//Step 5: Remove Persistent Read/Writes
			if(api == SystemMLAPI.JMLC) {
				//rewrite persistent reads/writes
				RewriteRemovePersistentReadWrite rewrite = new RewriteRemovePersistentReadWrite(inputs, outputs);
				ProgramRewriter rewriter2 = new ProgramRewriter(rewrite);
				rewriter2.rewriteProgramHopDAGs(prog);
			}
			else if(api == SystemMLAPI.MLContext) {
				//rewrite persistent reads/writes
				RewriteRemovePersistentReadWrite rewrite = new RewriteRemovePersistentReadWrite(inputs, outputs, symbolTable);
				ProgramRewriter rewriter2 = new ProgramRewriter(rewrite);
				rewriter2.rewriteProgramHopDAGs(prog);
			}

			//Step 6: construct lops (incl exec type and op selection)
			dmlt.constructLops(prog);

			if(DMLScript.LOG.isDebugEnabled()) {
				DMLScript.LOG.debug("\n********************** LOPS DAG *******************");
				dmlt.printLops(prog);
				dmlt.resetLopsDAGVisitStatus(prog);
			}

			//Step 7: generate runtime program, incl codegen
			rtprog = dmlt.getRuntimeProgram(prog, dmlconf);

			// Step 8: Cleanup/post-processing
			if(api == SystemMLAPI.JMLC) {
				JMLCUtils.cleanupRuntimeProgram(rtprog, outputs);
			}
			else if(api == SystemMLAPI.DMLScript) {
				//launch SystemML appmaster (if requested and not already in launched AM)
				if( dmlconf.getBooleanValue(DMLConfig.YARN_APPMASTER) ){
					if( !DMLScript.isActiveAM() && DMLYarnClientProxy.launchDMLYarnAppmaster(script, dmlconf, allArgs, rtprog) )
						return null; //if AM launch unsuccessful, fall back to normal execute
					if( DMLScript.isActiveAM() ) //in AM context (not failed AM launch)
						DMLAppMasterUtils.setupProgramMappingRemoteMaxMemory(rtprog);
				}
			}
			else if(api == SystemMLAPI.MLContext) {
				if (maintainSymbolTable) {
					MLContextUtil.deleteRemoveVariableInstructions(rtprog);
				} else {
					JMLCUtils.cleanupRuntimeProgram(rtprog, outputs);
				}
			}

			//Step 9: prepare statistics [and optional explain output]
			//count number compiled MR jobs / SP instructions
			if(api != SystemMLAPI.JMLC) {
				ExplainCounts counts = Explain.countDistributedOperations(rtprog);
				Statistics.resetNoOfCompiledJobs( counts.numJobs );
				//explain plan of program (hops or runtime)
				if( ConfigurationManager.getDMLOptions().explainType != ExplainType.NONE )
					System.out.println(
							Explain.display(prog, rtprog, ConfigurationManager.getDMLOptions().explainType, counts));

				Statistics.stopCompileTimer();
			}
		}
		catch(ParseException pe) {
			// don't chain ParseException (for cleaner error output)
			throw pe;
		}
		catch(Exception ex) {
			throw new DMLException(ex);
		}
		return rtprog;
	}

	/**
	 * Execute the runtime program. This involves execution of the program
	 * blocks that make up the runtime program and may involve dynamic
	 * recompilation.
	 *
	 * @param rtprog
	 *            runtime program
	 * @param statisticsMaxHeavyHitters
	 *            maximum number of statistics to print
	 * @param symbolTable
	 *            symbol table (that were registered as input as part of MLContext)
	 * @param outputVariables
	 *            output variables (that were registered as output as part of MLContext)
	 * @param api
	 * 			  API used to execute the runtime program
	 * @param gCtxs
	 * 			  list of GPU contexts
	 * @return execution context
	 */
	public static ExecutionContext executeRuntimeProgram(Program rtprog, int statisticsMaxHeavyHitters,
														 LocalVariableMap symbolTable, HashSet<String> outputVariables,
														 SystemMLAPI api, List<GPUContext> gCtxs) {
		boolean exceptionThrown = false;

		// Start timer
		Statistics.startRunTimer();

		// Create execution context and attach registered outputs
		ExecutionContext ec = ExecutionContextFactory.createContext(symbolTable, rtprog);
		if(outputVariables != null)
			ec.getVariables().setRegisteredOutputs(outputVariables);

		// Assign GPUContext to the current ExecutionContext
		if(gCtxs != null) {
			gCtxs.get(0).initializeThread();
			ec.setGPUContexts(gCtxs);
		}

		Exception finalizeException = null;
		try {
			// run execute (w/ exception handling to ensure proper shutdown)
			rtprog.execute(ec);
		} catch (Throwable e) {
			exceptionThrown = true;
			throw e;
		} finally { // ensure cleanup/shutdown
			if (ConfigurationManager.isGPU() && !ec.getGPUContexts().isEmpty()) {
				try {
					HashSet<MatrixObject> outputMatrixObjects = new HashSet<>();
					// -----------------------------------------------------------------
					// The below code pulls the output variables on the GPU to the host. This is required especially when:
					// The output variable was generated as part of a MLContext session with GPU enabled
					// and was passed to another MLContext with GPU disabled
					// The above scenario occurs in our gpu test suite (eg: BatchNormTest).
					if(outputVariables != null) {
						for(String outVar : outputVariables) {
							Data data = ec.getVariable(outVar);
							if(data instanceof MatrixObject) {
								for(GPUContext gCtx : ec.getGPUContexts()) {
									GPUObject gpuObj = ((MatrixObject)data).getGPUObject(gCtx);
									if(gpuObj != null && gpuObj.isDirty()) {
										gpuObj.acquireHostRead(null);
									}
								}
								outputMatrixObjects.add(((MatrixObject)data));
							}
						}
					}
					// -----------------------------------------------------------------
					for(GPUContext gCtx : ec.getGPUContexts()) {
						gCtx.clearTemporaryMemory(outputMatrixObjects);
					}
				} catch (Exception e1) {
					exceptionThrown = true;
					finalizeException = e1; // do not throw exception while cleanup
				}

			}
			if( ConfigurationManager.isCodegenEnabled() )
				SpoofCompiler.cleanupCodeGenerator();

			//cleanup unnecessary outputs
			if (outputVariables != null)
				symbolTable.removeAllNotIn(outputVariables);

			// Display statistics (disabled for JMLC)
			Statistics.stopRunTimer();
			if(api != SystemMLAPI.JMLC) {
				(exceptionThrown ? System.err : System.out)
						.println(Statistics.display(statisticsMaxHeavyHitters > 0 ?
								statisticsMaxHeavyHitters :
								ConfigurationManager.getDMLOptions().getStatisticsMaxHeavyHitters()));
			}
		}
		if(finalizeException != null) {
			throw new DMLRuntimeException("Error occured while GPU memory cleanup.", finalizeException);
		}
		return ec;
						}

}
