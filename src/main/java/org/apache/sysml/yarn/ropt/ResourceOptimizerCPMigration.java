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

package org.apache.sysml.yarn.ropt;


public class ResourceOptimizerCPMigration 
{
	
	/*// FIXME MB 
	private static final Log LOG = LogFactory.getLog(CPMigrationOptimizer.class.getName());
	
	public static void initResumeInfoFromFile(String file, ExecutionContext ec) throws IOException, DMLRuntimeException {
		DMLScript.resumeSbIdRStack.clear();
		DMLScript.resumeFuncVarRStack.clear();
		DMLScript.resumeLoopAndFuncEntryVarRStack.clear();
		if (file == null)
			return;
		
		FileSystem fs = FileSystem.get(new YarnConfiguration());
		Path path = new Path(file);
		if (!fs.exists(path))
			throw new IOException("File " + file + " does not exist");
		
		//BufferedReader reader = new BufferedReader(new InputStreamReader(fs.open(path)));
		FSDataInputStream reader = fs.open(path);
		DMLScript.predecessorAppIdStr = reader.readUTF().trim();
		DMLScript.resumeSbIdRStack = DMLScript.deserializeToReverseStack(reader.readUTF().trim());
		
		// Read and invert the function symbol table stack
		Stack<LocalVariableMap> tmpStack = new Stack<LocalVariableMap>();
		int n = Integer.parseInt(reader.readUTF().trim());
		for (int i = 0; i < n; i++)
			tmpStack.push(LocalVariableMap.deserialize(reader.readUTF().trim()));
		for (int i = 0; i < n; i++)
			DMLScript.resumeFuncVarRStack.push(tmpStack.pop());
		
		// Read and invert the loop and func entry symbol table stack
		tmpStack.clear();
		n = Integer.parseInt(reader.readUTF().trim());
		for (int i = 0; i < n; i++)
			tmpStack.push(LocalVariableMap.deserialize(reader.readUTF().trim()));
		for (int i = 0; i < n; i++)
			DMLScript.resumeLoopAndFuncEntryVarRStack.push(tmpStack.pop());
		
		reader.close();
		
		// Append this CP's AppId to the predecessor's resume file
		FSDataOutputStream fout = fs.append(path);
		fout.writeUTF(MyApplicationMaster.appId + "\n");
		fout.close();
		
		// Log the resume info
		StringBuilder sb = new StringBuilder();
		sb.append("Predecessor is " + DMLScript.predecessorAppIdStr + "\n");
		sb.append("Inverse sbID stack: ");
		for (Long l : DMLScript.resumeSbIdRStack)
			sb.append(l + ", ");
		sb.append("\n" + DMLScript.resumeFuncVarRStack.size() + " inverse function symbol table stack:\n");
		for (LocalVariableMap symbolTable : DMLScript.resumeFuncVarRStack)
			sb.append("\t" + symbolTable.serialize() + "\n");
		sb.append(DMLScript.resumeLoopAndFuncEntryVarRStack.size() + " inverse loop and func entry symbol table stack:\n");
		for (LocalVariableMap symbolTable : DMLScript.resumeLoopAndFuncEntryVarRStack)
			sb.append("\t" + symbolTable.serialize() + "\n");
		LOG.info(sb.toString());
		
		// Recover the most outer layer of symbol table before execution starts
		LocalVariableMap varMap = DMLScript.resumeFuncVarRStack.pop();
		DMLScript.lastPopedFuncVarStr = varMap.serialize();	// For later correctness check
		ec.setVariables(varMap);
	}
	
	// Try to migrate to a new CP, return false if failed
	public static boolean migrateCP(LocalVariableMap currentSymbolTable) throws DMLRuntimeException {
		DMLScript.execFuncVarStack.push(currentSymbolTable);
		
		long start = System.currentTimeMillis();
		for (LocalVariableMap symbolTable : DMLScript.execFuncVarStack) {
			for (String var : symbolTable.keySet()) {
				Data data = symbolTable.get(var);
				if ( data.getDataType() == DataType.MATRIX ) {
					long time = System.currentTimeMillis();
					MatrixObject matrix = (MatrixObject) data;
					matrix.exportData();
					time = System.currentTimeMillis() - time;
					LOG.info("Exporting " + var + " took " + time + "ms");
				}
			}
		}
		start = System.currentTimeMillis() - start;
		LOG.info("Exporting data to hdfs took " + start + "ms");
		
		DMLConfig config = ConfigurationManager.getConfig();
		String hdfsWorkingDir = MyYarnClient.getHDFSWorkingDir(config, MyApplicationMaster.appId);
		
		try {
			FileSystem fs = FileSystem.get(MyApplicationMaster.conf);
			Path resume_file = new Path(hdfsWorkingDir, DMLScript.CP_RESUME_HDFS_FILE);
			if (fs.exists(resume_file))
				throw new IOException("File " + resume_file + " already exists?");
			
			FSDataOutputStream fout = fs.create(resume_file);
			fout.writeUTF(MyApplicationMaster.appId + "\n");
			
			String tmp = DMLScript.serializeExecSbIdStack(DMLScript.execSbIdStack);
			fout.writeUTF(tmp + "\n");
			LOG.info("SbId Stack: " + tmp);
			
			fout.writeUTF(DMLScript.execFuncVarStack.size() + "\n");
			LOG.info(DMLScript.execFuncVarStack.size() + " func symbol tables");
			for (LocalVariableMap symbolTable : DMLScript.execFuncVarStack) {
				tmp = symbolTable.serialize();
				fout.writeUTF(symbolTable.serialize() + "\n");
				LOG.info("\t" + tmp);
			}
			
			fout.writeUTF(DMLScript.execLoopAndFuncEntryVarStack.size() + "\n");
			LOG.info(DMLScript.execLoopAndFuncEntryVarStack.size() + " loop and func entry symbol tables");
			for (LocalVariableMap symbolTable : DMLScript.execLoopAndFuncEntryVarStack) {
				tmp = symbolTable.serialize();
				fout.writeUTF(symbolTable.serialize() + "\n");
				LOG.info("\t" + tmp);
			}
			fout.close();
			
			// If specified, run the new CP within this old CP automatically
			if (DMLScript.newCpBudgetByte != -1) {
				YarnApplicationState state;
				MyYarnClient client = new MyYarnClient();
				state = client.runResumeCP(MyApplicationMaster.fullArgs, DMLScript.newCpBudgetByte, hdfsWorkingDir);
				if (state != YarnApplicationState.FINISHED)
					throw new DMLRuntimeException("Resuming CP " + state);
			}
		} catch (Exception e) {
			throw new DMLRuntimeException(e);
		}
		
		// Make this CP stop gracefully
		DMLScript.migratedAndStop = true;
		
		LOG.info("CP migration done, exiting gracefully");
		return true;
	}
	
	
	
	
	
	// A complete full copy of the runtime plan for recompile and costing
	public ArrayList<ProgramBlock> copiedProgramBlocks;
	Program prog;
	
	public HashMap<Long, ProgramBlock> sbIdMap;		// Pointer to all reachable program blocks
	
	public CPMigrationOptimizer(Program rtprog) throws DMLRuntimeException, HopsException {
		prog = rtprog;
		copiedProgramBlocks = ProgramConverter.rcreateDeepCopyProgramBlocks(prog._programBlocks, 1, -1, new HashSet<String>(), false);
		
		// Temporarily disable dynamic recompile to clear all flags
		boolean flag = OptimizerUtils.ALLOW_DYN_RECOMPILATION;
		OptimizerUtils.ALLOW_DYN_RECOMPILATION = false;
		
		sbIdMap = new HashMap<Long, ProgramBlock> ();
		for (ProgramBlock pb : copiedProgramBlocks)
			initTraverse(pb);
		OptimizerUtils.ALLOW_DYN_RECOMPILATION = flag;
		
		
	}
	
	public void recompile() throws HopsException {
		resetAllFlags();
	}
	
	public boolean shouldMigrate(Stack<Long> execSbIdStack, Stack<LocalVariableMap> execLoopAndFuncEntryVarStack) 
			throws HopsException {
		
		int i = 0, j = 0;
		int base = -1;	// Index of base function, -1 for main
		LocalVariableMap baseVar = new LocalVariableMap();
		int loop = -1;	// Index of the outer most loop
		HashMap<Long, LocalVariableMap> loopEntryVarMap = new HashMap<Long, LocalVariableMap>();
		
		for (Long sbId : execSbIdStack) {
			ProgramBlock pb = sbIdMap.get(sbId);
			if (pb instanceof FunctionProgramBlock) {
				base = i;
				baseVar = execLoopAndFuncEntryVarStack.get(j);
				loop = -1;
				loopEntryVarMap.clear();
				j++;
			} else if (pb instanceof WhileProgramBlock || pb instanceof ForProgramBlock) {
				if (loop == -1)
					loop = i;
				loopEntryVarMap.put(sbId, execLoopAndFuncEntryVarStack.get(j));
				j++;
			}
			i++;
		}
		
		// To be continued !!!!!
		
		return false;
	}
	
	public void resetAllFlags() throws HopsException {
		boolean flag = OptimizerUtils.ALLOW_DYN_RECOMPILATION;
		OptimizerUtils.ALLOW_DYN_RECOMPILATION = false;
		for (Map.Entry<Long, ProgramBlock> entry : sbIdMap.entrySet())
			entry.getValue().getStatementBlock().updateRecompilationFlag();
		OptimizerUtils.ALLOW_DYN_RECOMPILATION = flag;
	}
	
	public void initTraverse(ProgramBlock pb) throws HopsException, DMLRuntimeException {
		long sbId = pb.getStatementBlock().getID();
		if (sbIdMap.containsKey(sbId))
			return;
		
		sbIdMap.put(sbId, pb);
		if (pb instanceof WhileProgramBlock) {
			WhileProgramBlock tmp = (WhileProgramBlock)pb;
			for (ProgramBlock pb2 : tmp.getChildBlocks())
				initTraverse(pb2);
		} else if (pb instanceof IfProgramBlock) {
			IfProgramBlock tmp = (IfProgramBlock)pb;
			for (ProgramBlock pb2 : tmp.getChildBlocksIfBody())
				initTraverse(pb2);
			for (ProgramBlock pb2 : tmp.getChildBlocksElseBody())
				initTraverse(pb2);
		} else if (pb instanceof ForProgramBlock) {
			ForProgramBlock tmp = (ForProgramBlock)pb;	
			for (ProgramBlock pb2 : tmp.getChildBlocks())
				initTraverse(pb2);
		} else if (pb instanceof FunctionProgramBlock && !(pb instanceof ExternalFunctionProgramBlock)) {
			FunctionProgramBlock tmp = (FunctionProgramBlock) pb;
			for (ProgramBlock pb2 : tmp.getChildBlocks())
				initTraverse(pb2);
		} else {
			// Clear the flag on leaf blocks
			pb.getStatementBlock().updateRecompilationFlag();
			for (Instruction inst : pb.getInstructions()) {
				if (inst instanceof FunctionCallCPInstruction) {
					FunctionCallCPInstruction finst = (FunctionCallCPInstruction)inst;
					initTraverse(prog.getFunctionProgramBlock(finst.getNamespace(), finst.getFunctionName()));
				}
			}
		}
	}
	
	*/
}
