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

package org.apache.sysds.test.component.parfor;

import static org.junit.Assert.fail;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.controlprogram.ParForProgramBlock;
import org.apache.sysds.runtime.controlprogram.ParForProgramBlock.PResultMerge;
import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContextFactory;
import org.apache.sysds.runtime.controlprogram.parfor.ResultMerge;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.meta.MetaDataFormat;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public class ResultMergeTest extends AutomatedTestBase{
	private final static String TEST_NAME = "parfor_rm";
	private final static String TEST_DIR = "functions/parfor/";
	private static final String TEST_CLASS_DIR = TEST_DIR + ResultMergeTest.class.getSimpleName() + "/";
	private static final String PACKAGE = "org.apache.sysds.runtime.controlprogram.parfor";
	private static Level _oldLevel = null;
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"C"}));
		_oldLevel = Logger.getLogger(PACKAGE).getLevel();
		Logger.getLogger(PACKAGE).setLevel( Level.TRACE );
	}
	
	@Override
	public void tearDown() {
		super.tearDown();
		Logger.getLogger(PACKAGE).setLevel( _oldLevel );
	}
	
	@Test
	public void testLocalMemDenseCompare() {
		testResultMergeAll(PResultMerge.LOCAL_MEM, false);
	}
	
	@Test
	public void testLocalFileDenseCompare() {
		testResultMergeAll(PResultMerge.LOCAL_FILE, false);
	}
	
	@Test
	public void testLocalAutomaticDenseCompare() {
		testResultMergeAll(PResultMerge.LOCAL_AUTOMATIC, false);
	}
	
	@Test
	public void testLocalMemSparseCompare() {
		testResultMergeAll(PResultMerge.LOCAL_MEM, true);
	}
	
	@Test
	public void testLocalFileSparseCompare() {
		testResultMergeAll(PResultMerge.LOCAL_FILE, true);
	}
	
	@Test
	public void testLocalAutomaticSparseCompare() {
		testResultMergeAll(PResultMerge.LOCAL_AUTOMATIC, true);
	}
	
	private void testResultMergeAll(PResultMerge mtype, boolean sparseCompare) {
		testResultMerge(false, false, false, sparseCompare, mtype);
		testResultMerge(false, true, false, sparseCompare, mtype);
		testResultMerge(true, false, false, sparseCompare, mtype);
		testResultMerge(false, false, true, sparseCompare, mtype);
		testResultMerge(false, true, true, sparseCompare, mtype);
		testResultMerge(true, false, true, sparseCompare, mtype);
		//testResultMerge(true, true, false, false, mtype); invalid
	}
	
	private void testResultMerge(boolean par, boolean accum, boolean compare, boolean sparseCompare, PResultMerge mtype) {
		try{
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
	
			//create input and output objects
			int rows = 1200, cols = 1100;
			MatrixBlock A = MatrixBlock.randOperations(rows,cols,0.1,0,1,"uniform",7);
			A.checkSparseRows();
			MatrixBlock rest = compare ? 
				MatrixBlock.randOperations(rows/3,cols,sparseCompare?0.2:1.0,1,1,"uniform",3): //constant
				new MatrixBlock(rows/3,cols,true); //empty (also sparse)
			CacheableData<?> Cobj = compare ?
				toMatrixObject(rest.rbind(rest).rbind(rest), output("C")) :
				toMatrixObject(new MatrixBlock(rows,cols,true), output("C"));
			MatrixObject[] Bobj = new MatrixObject[3];
			Bobj[0] = toMatrixObject(A.slice(0,rows/3-1).rbind(rest).rbind(rest), output("B0"));
			Bobj[1] = toMatrixObject(rest.rbind(A.slice(rows/3,2*rows/3-1)).rbind(rest), output("B1"));
			Bobj[2] = toMatrixObject(rest.rbind(rest).rbind(A.slice(2*rows/3,rows-1)), output("B2"));
			BinaryOperator PLUS = InstructionUtils.parseBinaryOperator("+");
			BinaryOperator MINUS = InstructionUtils.parseBinaryOperator("-");
			MatrixBlock aggAll = ((MatrixBlock)Cobj.acquireReadAndRelease())
				.binaryOperations(PLUS, Bobj[0].acquireReadAndRelease())
				.binaryOperations(PLUS, Bobj[1].acquireReadAndRelease())
				.binaryOperations(PLUS, Bobj[2].acquireReadAndRelease())
				.binaryOperations(MINUS, (MatrixBlock)Cobj.acquireReadAndRelease())
				.binaryOperations(MINUS, (MatrixBlock)Cobj.acquireReadAndRelease())
				.binaryOperations(MINUS, (MatrixBlock)Cobj.acquireReadAndRelease());
			
			//create result merge
			ExecutionContext ec = ExecutionContextFactory.createContext();
			int numThreads = 4;
			ResultMerge<?> rm = ParForProgramBlock.createResultMerge(
				mtype, Cobj, Bobj, output("R"), accum, numThreads, ec);
				
			//execute results merge
			if( par )
				Cobj = rm.executeParallelMerge(numThreads);
			else 
				Cobj = rm.executeSerialMerge();
			
			//check results
			MatrixBlock out = (MatrixBlock)Cobj.acquireReadAndRelease();
			if(!accum)
				TestUtils.compareMatrices(A, out, 1e-14);
			else
				TestUtils.compareMatrices(aggAll, out, 1e-14);
		}
		catch(Exception e){
			e.printStackTrace();
			fail(e.getMessage());
		}
	}
	
	private static MatrixObject toMatrixObject(MatrixBlock mb, String filename) {
		MetaDataFormat md = new MetaDataFormat(
			mb.getDataCharacteristics().setBlocksize(1000), FileFormat.BINARY);
		MatrixObject mo = new MatrixObject(ValueType.FP64, filename, md);
		mo.acquireModify(mb);
		mo.release();
		return mo;
	}
}
