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

package org.apache.sysds.test.component.compress.lib;

import static org.junit.Assert.fail;

import java.util.List;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.log4j.spi.LoggingEvent;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressedMatrixBlockFactory;
import org.apache.sysds.runtime.compress.lib.CLALibBinaryCellOp;
import org.apache.sysds.runtime.functionobjects.GreaterThan;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.test.LoggingUtils;
import org.apache.sysds.test.LoggingUtils.TestAppender;
import org.junit.Test;

public class CLALibBinaryCellOpLoggingTest {
	
	public CLALibBinaryCellOpLoggingTest(){
		// enable statistics.
		DMLScript.STATISTICS = true;
	}


	@Test 
	public void binaryOpLoggingTest_TRACE(){
		final TestAppender appender = LoggingUtils.overwrite();

		try {
			Logger.getLogger(CLALibBinaryCellOp.class).setLevel(Level.TRACE);
			DMLScript.STATISTICS = true;
			CompressedMatrixBlock cmb = CompressedMatrixBlockFactory.createConstant(100, 10, 324);
			MatrixBlock m2 = new MatrixBlock(100, 1, 324.2);
			BinaryOperator op = new BinaryOperator(Plus.getPlusFnObject(), 2);
		
			CLALibBinaryCellOp.binaryOperationsRight(op, cmb, m2);
			final List<LoggingEvent> log = LoggingUtils.reinsert(appender);
			for(LoggingEvent l : log) {
				if(l.getMessage().toString().contains("decompressed block w/ k"))
					return;
			}
			fail("decompressed block ");
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
		finally {
			Logger.getLogger(CLALibBinaryCellOp.class).setLevel(Level.WARN);
			LoggingUtils.reinsert(appender);
		}
	}


	@Test 
	public void binaryOpLoggingTest_DEBUG(){
		final TestAppender appender = LoggingUtils.overwrite();

		try {
			Logger.getLogger(CLALibBinaryCellOp.class).setLevel(Level.DEBUG);
			DMLScript.STATISTICS = true;
			CompressedMatrixBlock cmb = CompressedMatrixBlockFactory.createConstant(100, 10, 324);
			MatrixBlock m2 = new MatrixBlock(100, 1, 324.2);
			BinaryOperator op = new BinaryOperator(Plus.getPlusFnObject(), 2);
		
			CLALibBinaryCellOp.binaryOperationsRight(op, cmb, m2);
			final List<LoggingEvent> log = LoggingUtils.reinsert(appender);
			for(LoggingEvent l : log) {
				if(l.getMessage().toString().contains("decompressed block w/ k"))
					fail("decompressed block ");
					
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
		finally {
			Logger.getLogger(CLALibBinaryCellOp.class).setLevel(Level.WARN);
			LoggingUtils.reinsert(appender);
		}
	}


	@Test 
	public void binaryOpLoggingTest_sparseOut_TRACE(){
		final TestAppender appender = LoggingUtils.overwrite();

		try {
			Logger.getLogger(CLALibBinaryCellOp.class).setLevel(Level.TRACE);
			DMLScript.STATISTICS = true;
			CompressedMatrixBlock cmb = CompressedMatrixBlockFactory.createConstant(100, 10, 324);
			MatrixBlock m2 = new MatrixBlock(100, 1, 324.2);
			BinaryOperator op = new BinaryOperator(GreaterThan.getGreaterThanFnObject(), 2);
		
			CLALibBinaryCellOp.binaryOperationsRight(op, cmb, m2);
			final List<LoggingEvent> log = LoggingUtils.reinsert(appender);
			for(LoggingEvent l : log) {
				if(l.getMessage().toString().contains("decompressed block w/ k"))
					return;
			}
			fail("decompressed block ");
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
		finally {
			Logger.getLogger(CLALibBinaryCellOp.class).setLevel(Level.WARN);
			LoggingUtils.reinsert(appender);
		}
	}


	@Test 
	public void binaryOpLoggingTest_sparseOut_DEBUG(){
		final TestAppender appender = LoggingUtils.overwrite();

		try {
			Logger.getLogger(CLALibBinaryCellOp.class).setLevel(Level.DEBUG);
			DMLScript.STATISTICS = true;
			CompressedMatrixBlock cmb = CompressedMatrixBlockFactory.createConstant(100, 10, 324);
			MatrixBlock m2 = new MatrixBlock(100, 1, 324.2);
			BinaryOperator op = new BinaryOperator(GreaterThan.getGreaterThanFnObject(), 2);
		
			CLALibBinaryCellOp.binaryOperationsRight(op, cmb, m2);
			final List<LoggingEvent> log = LoggingUtils.reinsert(appender);
			for(LoggingEvent l : log) {
				if(l.getMessage().toString().contains("decompressed block w/ k"))
					fail("decompressed block ");
					
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
		finally {
			Logger.getLogger(CLALibBinaryCellOp.class).setLevel(Level.WARN);
			LoggingUtils.reinsert(appender);
		}
	}

}
