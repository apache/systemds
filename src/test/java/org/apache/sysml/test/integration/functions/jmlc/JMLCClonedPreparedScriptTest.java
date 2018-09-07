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

package org.apache.sysml.test.integration.functions.jmlc;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysml.api.DMLException;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.jmlc.Connection;
import org.apache.sysml.api.jmlc.PreparedScript;
import org.apache.sysml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.utils.Statistics;

public class JMLCClonedPreparedScriptTest extends AutomatedTestBase 
{
	//basic script with parfor loop
	private static final String SCRIPT1 =
		  "X = matrix(7, 10, 10);"
		+ "R = matrix(0, 10, 1)"
		+ "parfor(i in 1:nrow(X))"
		+ "  R[i,] = sum(X[i,])"
		+ "out = sum(R)"
		+ "write(out, 'tmp/out')";
	
	//script with dml-bodied and external functions
	private static final String SCRIPT2 =
		  "foo1 = externalFunction(int numInputs, boolean stretch, Matrix[double] A, Matrix[double] B, Matrix[double] C) "
		+ "  return (Matrix[double] D)"
		+ "  implemented in (classname='org.apache.sysml.udf.lib.MultiInputCbind', exectype='mem');"
		+ "foo2 = function(Matrix[double] A, Matrix[double] B, Matrix[double] C)"
		+ "  return (Matrix[double] D) {"
		+ "  while(FALSE){}"
		+ "  D = cbind(A, B, C)"
		+ "}"
		+ "X = matrix(7, 10, 10);"
		+ "R = matrix(0, 10, 1)"
		+ "for(i in 1:nrow(X)) {"
		+ "  D = foo1(3, FALSE, X[i,], X[i,], X[i,])"
		+ "  E = foo2(D, D, D)"
		+ "  R[i,] = sum(E)/9"
		+ "}"
		+ "out = sum(R)"
		+ "write(out, 'tmp/out')";
	
	
	@Override
	public void setUp() {
		//do nothing
	}
	
	@Test
	public void testSinglePreparedScript1T128() throws IOException {
		runJMLCClonedTest(SCRIPT1, 128, false);
	}
	
	@Test
	public void testClonedPreparedScript1T128() throws IOException {
		runJMLCClonedTest(SCRIPT1, 128, true);
	}
	
	@Test
	public void testSinglePreparedScript2T128() throws IOException {
		runJMLCClonedTest(SCRIPT2, 128, false);
	}
	
	@Test
	public void testClonedPreparedScript2T128() throws IOException {
		runJMLCClonedTest(SCRIPT2, 128, true);
	}

	private void runJMLCClonedTest(String script, int num, boolean clone) 
		throws IOException
	{
		int k = InfrastructureAnalyzer.getLocalParallelism();
		
		boolean failed = false;
		try( Connection conn = new Connection() ) {
			DMLScript.STATISTICS = true;
			Statistics.reset();
			PreparedScript pscript = conn.prepareScript(
				script, new String[]{}, new String[]{"out"}, false);
			
			ExecutorService pool = Executors.newFixedThreadPool(k);
			ArrayList<JMLCTask> tasks = new ArrayList<>();
			for(int i=0; i<num; i++)
				tasks.add(new JMLCTask(pscript, clone));
			List<Future<Double>> taskrets = pool.invokeAll(tasks);
			for(Future<Double> ret : taskrets)
				if( ret.get() != 700 )
					throw new RuntimeException("wrong results: "+ret.get());
			pool.shutdown();
		}
		catch(Exception ex) {
			failed = true;
		}
		
		//check expected failure
		Assert.assertTrue(failed==!clone || k==1);
	}
	
	private static class JMLCTask implements Callable<Double> 
	{
		private final PreparedScript _pscript;
		private final boolean _clone;
		
		protected JMLCTask(PreparedScript pscript, boolean clone) {
			_pscript = pscript;
			_clone = clone;
		}
		
		@Override
		public Double call() throws DMLException
		{
			if( _clone )
				return _pscript.clone(false).executeScript().getDouble("out");
			else
				return _pscript.executeScript().getDouble("out");
		}
	}
}
