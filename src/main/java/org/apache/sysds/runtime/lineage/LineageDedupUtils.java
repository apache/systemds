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

package org.apache.sysds.runtime.lineage;

import org.apache.sysds.runtime.controlprogram.ForProgramBlock;
import org.apache.sysds.runtime.controlprogram.FunctionProgramBlock;
import org.apache.sysds.runtime.controlprogram.IfProgramBlock;
import org.apache.sysds.runtime.controlprogram.ProgramBlock;
import org.apache.sysds.runtime.controlprogram.WhileProgramBlock;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;

public class LineageDedupUtils {
	
	public static boolean isValidDedupBlock(ProgramBlock pb, boolean inLoop) {
		boolean ret = true; //basic program block
		if (pb instanceof FunctionProgramBlock) {
			FunctionProgramBlock fsb = (FunctionProgramBlock)pb;
			for (ProgramBlock cpb : fsb.getChildBlocks())
				ret &= isValidDedupBlock(cpb, inLoop);
		}
		else if (pb instanceof WhileProgramBlock) {
			if( inLoop ) return false;
			WhileProgramBlock wpb = (WhileProgramBlock) pb;
			for (ProgramBlock cpb : wpb.getChildBlocks())
				ret &= isValidDedupBlock(cpb, true);
		}
		else if (pb instanceof IfProgramBlock) {
			IfProgramBlock ipb = (IfProgramBlock) pb;
			for (ProgramBlock cpb : ipb.getChildBlocksIfBody())
				ret &= isValidDedupBlock(cpb, inLoop);
			for (ProgramBlock cpb : ipb.getChildBlocksElseBody())
				ret &= isValidDedupBlock(cpb, inLoop);
		}
		else if (pb instanceof ForProgramBlock) { //incl parfor
			if( inLoop ) return false;
			ForProgramBlock fpb = (ForProgramBlock) pb;
			for (ProgramBlock cpb : fpb.getChildBlocks())
				ret &= isValidDedupBlock(cpb, true);
		}
		return ret;
	}
	
	public static LineageDedupBlock computeDedupBlock(ProgramBlock fpb, ExecutionContext ec) {
		LineageDedupBlock ldb = new LineageDedupBlock();
		ec.getLineage().setInitDedupBlock(ldb);
		ldb.traceProgramBlocks(fpb.getChildBlocks(), ec);
		ec.getLineage().setInitDedupBlock(null);
		return ldb;
	}
}
