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

import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.controlprogram.ParForProgramBlock;
import org.apache.sysds.runtime.controlprogram.ParForProgramBlock.PResultMerge;
import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContextFactory;
import org.apache.sysds.runtime.controlprogram.parfor.ResultMerge;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.meta.MetaDataFormat;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Ignore;
import org.junit.Test;

public class ResultMergeTest extends AutomatedTestBase{
	private final static String TEST_NAME = "parfor_rm";
	private final static String TEST_DIR = "functions/parfor/";
	private static final String TEST_CLASS_DIR = TEST_DIR + ResultMergeTest.class.getSimpleName() + "/";
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"C"})); 
	}
	
	@Test
	public void testLocalMem() {
		testResultMergeAll(PResultMerge.LOCAL_MEM);
	}
	
	@Test
	@Ignore //FIXME
	public void testLocalFile() {
		testResultMergeAll(PResultMerge.LOCAL_FILE);
	}
	
	@Test
	public void testLocalAutomatic() {
		testResultMergeAll(PResultMerge.LOCAL_AUTOMATIC);
	}
	
	private void testResultMergeAll(PResultMerge mtype) {
		testResultMerge(false, false, false, mtype);
		testResultMerge(false, true, false, mtype);
		testResultMerge(true, false, false, mtype);
		//testResultMerge(true, true, false, mtype); invalid
	}
	
	private void testResultMerge(boolean par, boolean accum, boolean compare, PResultMerge mtype) {
		loadTestConfiguration(getTestConfiguration(TEST_NAME));

		//create input and output objects
		MatrixBlock A = MatrixBlock.randOperations(1200, 1100, 0.1);
		CacheableData<?> Cobj = compare ?
			toMatrixObject(A, output("C")) :
			toMatrixObject(new MatrixBlock(1200,1100,true), output("C"));
		MatrixBlock empty = new MatrixBlock(400,1100,true);
		MatrixObject[] Bobj = new MatrixObject[3];
		Bobj[0] = toMatrixObject(A.slice(0,399).rbind(empty).rbind(empty), output("B0"));
		Bobj[1] = toMatrixObject(empty.rbind(A.slice(400,799)).rbind(empty), output("B1"));
		Bobj[2] = toMatrixObject(empty.rbind(empty).rbind(A.slice(800,1199)), output("B1"));
		
		//create result merge
		ExecutionContext ec = ExecutionContextFactory.createContext();
		int numThreads = 3;
		ResultMerge<?> rm = ParForProgramBlock.createResultMerge(
			mtype, Cobj, Bobj, output("C"), accum, numThreads, ec);
			
		//execute results merge
		if( par )
			Cobj = rm.executeParallelMerge(numThreads);
		else 
			Cobj = rm.executeSerialMerge();
		
		//check results
		TestUtils.compareMatrices(A, 
			(MatrixBlock)Cobj.acquireReadAndRelease(), 1e-14);
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
