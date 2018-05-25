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

package org.apache.sysml.runtime.controlprogram.paramserv;

import java.util.ArrayList;

import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.hops.DataOp;
import org.apache.sysml.hops.Hop;
import org.apache.sysml.hops.IndexingOp;
import org.apache.sysml.hops.LiteralOp;
import org.apache.sysml.hops.rewrite.HopRewriteUtils;
import org.apache.sysml.lops.Lop;
import org.apache.sysml.lops.compile.Dag;
import org.apache.sysml.parser.DataIdentifier;
import org.apache.sysml.parser.Expression;
import org.apache.sysml.runtime.controlprogram.Program;
import org.apache.sysml.runtime.controlprogram.ProgramBlock;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContextFactory;
import org.apache.sysml.runtime.instructions.Instruction;
import org.apache.sysml.runtime.instructions.cp.Data;
import org.apache.sysml.runtime.instructions.cp.ListObject;

public class ParamservUtils {

	public static void populate(ExecutionContext ec, ListObject lo) {
		lo.getNames().forEach(name -> {
			Data newData = lo.slice(name);
			ec.setVariable(name, newData);
		});
	}

	public static MatrixObject sliceMatrix(MatrixObject mo, long rl, long rh) {
		Hop inHop = new DataOp("in", Expression.DataType.MATRIX, Expression.ValueType.DOUBLE,
				Hop.DataOpTypes.TRANSIENTREAD, mo.getFileName(), mo.getNumRows(), mo.getNumRows(), mo.getNnz(),
				(int) mo.getNumRowsPerBlock(), (int) mo.getNumColumnsPerBlock());
		IndexingOp iop = HopRewriteUtils.createIndexingOp(inHop, new LiteralOp(rl), new LiteralOp(rh), new LiteralOp(1),
				new LiteralOp(mo.getNumColumns()));
		Hop outHop = new DataOp("out", Expression.DataType.MATRIX, Expression.ValueType.DOUBLE, iop,
				Hop.DataOpTypes.TRANSIENTWRITE, iop.getFilename());

		//generate runtime instruction
		Dag<Lop> dag = new Dag<>();
		Lop lops = outHop.constructLops(); //reconstruct lops
		lops.addToDag(dag);
		ArrayList<Instruction> inst = dag.getJobs(null, ConfigurationManager.getDMLConfig());

		//execute instructions
		ExecutionContext ec = ExecutionContextFactory.createContext();
		ec.setVariable("in", mo);
		ProgramBlock pb = new ProgramBlock(new Program());
		pb.setInstructions(inst);
		pb.execute(ec);

		MatrixObject out = ec.getMatrixObject("out");

		//clean up
		pb.setInstructions(null);
		ec.getVariables().removeAll();

		return out;
	}

	public static void populate(ExecutionContext ec, ListObject hyperParams, ArrayList<DataIdentifier> inputs) {
		inputs.forEach(input -> {
			String name = input.getName();
			if (hyperParams.getNames().contains(name)) {
				ec.setVariable(input.getName(), hyperParams.slice(name, input.getValueType(), input.getDataType()));
			}
		});
	}
}
