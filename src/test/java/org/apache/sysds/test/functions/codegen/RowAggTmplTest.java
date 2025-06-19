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

package org.apache.sysds.test.functions.codegen;

import java.io.File;
import java.util.HashMap;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Ignore;
import org.junit.Test;

public class RowAggTmplTest extends AutomatedTestBase
{
	private static final Log LOG = LogFactory.getLog(RowAggTmplTest.class.getName());

	private static final String TEST_NAME = "rowAggPattern";
	private static final String TEST_NAME1 = TEST_NAME+"1"; //t(X)%*%(X%*%(lamda*v))
	private static final String TEST_NAME2 = TEST_NAME+"2"; //t(X)%*%(lamda*(X%*%v))
	private static final String TEST_NAME3 = TEST_NAME+"3"; //t(X)%*%(z+(2-(w*(X%*%v))))
	private static final String TEST_NAME4 = TEST_NAME+"4"; //colSums(X/rowSums(X))
	private static final String TEST_NAME5 = TEST_NAME+"5"; //t(X)%*%((P*(1-P))*(X%*%v));
	private static final String TEST_NAME6 = TEST_NAME+"6"; //t(X)%*%((P[,1]*(1-P[,1]))*(X%*%v));
	private static final String TEST_NAME7 = TEST_NAME+"7"; //t(X)%*%(X%*%v-y); sum((X%*%v-y)^2);
	private static final String TEST_NAME8 = TEST_NAME+"8"; //colSums((X/rowSums(X))>0.7)
	private static final String TEST_NAME9 = TEST_NAME+"9"; //t(X) %*% (v - abs(y))
	private static final String TEST_NAME10 = TEST_NAME+"10"; //Y=(X<=rowMins(X)); R=colSums((Y/rowSums(Y)));
	private static final String TEST_NAME11 = TEST_NAME+"11"; //y - X %*% v
	private static final String TEST_NAME12 = TEST_NAME+"12"; //Y=(X>=v); R=Y/rowSums(Y)
	private static final String TEST_NAME13 = TEST_NAME+"13"; //rowSums(X)+rowSums(Y)
	private static final String TEST_NAME14 = TEST_NAME+"14"; //colSums(max(floor(round(abs(min(sign(X+Y),rowSums(X))))),7))
	private static final String TEST_NAME15 = TEST_NAME+"15"; //systemds nn - softmax backward
	private static final String TEST_NAME16 = TEST_NAME+"16"; //Y=X-rowIndexMax(X); R=Y/rowSums(Y)
	private static final String TEST_NAME17 = TEST_NAME+"17"; //MLogreg - vector-matrix w/ indexing
	private static final String TEST_NAME18 = TEST_NAME+"18"; //MLogreg - matrix-vector cbind 0s
	private static final String TEST_NAME19 = TEST_NAME+"19"; //MLogreg - rowwise dag
	private static final String TEST_NAME20 = TEST_NAME+"20"; //1 / (1 - (A / rowSums(A)))
	private static final String TEST_NAME21 = TEST_NAME+"21"; //sum(X/rowSums(X))
	private static final String TEST_NAME22 = TEST_NAME+"22"; //((7+X)+(X-7)+exp(X))/(rowMins(X)+0.5) 
	private static final String TEST_NAME23 = TEST_NAME+"23"; //L2SVM outer loop 
	private static final String TEST_NAME24 = TEST_NAME+"24"; //t(X)%*%(w*(X%*%v)), w/ mm 
	private static final String TEST_NAME25 = TEST_NAME+"25"; //-2*(X%*%t(C))+t(rowSums(C^2)), w/ mm
	private static final String TEST_NAME26 = TEST_NAME+"26"; //t(P)%*%X, w/ mm
	private static final String TEST_NAME27 = TEST_NAME+"27"; //t(X)%*%(X%*%v), w/ mm 
	private static final String TEST_NAME28 = TEST_NAME+"28"; //Kmeans, final eval
	private static final String TEST_NAME29 = TEST_NAME+"29"; //sum(rowMins(X))
	private static final String TEST_NAME30 = TEST_NAME+"30"; //Mlogreg inner core, multi-class
	private static final String TEST_NAME31 = TEST_NAME+"31"; //MLogreg - matrix-vector cbind 0s generalized
	private static final String TEST_NAME32 = TEST_NAME+"32"; //X[, 1] - rowSums(X)
	private static final String TEST_NAME33 = TEST_NAME+"33"; //Kmeans, inner loop
	private static final String TEST_NAME34 = TEST_NAME+"34"; //X / rowSums(X!=0)
	private static final String TEST_NAME35 = TEST_NAME+"35"; //cbind(X/rowSums(X), Y, Z)
	private static final String TEST_NAME36 = TEST_NAME+"36"; //xor operation
	private static final String TEST_NAME37 = TEST_NAME+"37"; //sprop(X/rowSums)
	private static final String TEST_NAME38 = TEST_NAME+"38"; //sigmoid(X/rowSums)
	private static final String TEST_NAME39 = TEST_NAME+"39"; //BitwAnd operation
	private static final String TEST_NAME40 = TEST_NAME+"40"; //relu operation -> (X>0)* dout
	private static final String TEST_NAME41 = TEST_NAME+"41"; //X*rowSums(X/seq(1,N)+t(seq(M,1)))
	private static final String TEST_NAME42 = TEST_NAME+"42"; //X/rowSums(min(X, Y, Z))
	private static final String TEST_NAME43 = TEST_NAME+"43"; //bias_add(X,B) + bias_mult(X,B)
	private static final String TEST_NAME44 = TEST_NAME+"44"; //maxpool(X - mean(X)) + 7;
	private static final String TEST_NAME45 = TEST_NAME+"45"; //vector allocation;
	private static final String TEST_NAME46 = TEST_NAME+"46"; //conv2d(X - mean(X), F1) + conv2d(X - mean(X), F2);
	private static final String TEST_NAME47 = TEST_NAME+"47"; //sum(X + rowVars(X))
	private static final String TEST_NAME48 = TEST_NAME+"48"; //sum(rowVars(X))
	private static final String TEST_NAME49 = TEST_NAME+"49";

	private static final String TEST_DIR = "functions/codegen/";
	private static final String TEST_CLASS_DIR = TEST_DIR + RowAggTmplTest.class.getSimpleName() + "/";
	private final static String TEST_CONF = "SystemDS-config-codegen.xml";
	private final static File   TEST_CONF_FILE = new File(SCRIPT_DIR + TEST_DIR, TEST_CONF);

	private static double eps = Math.pow(10, -10);
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		for(int i=1; i<=49; i++)
			addTestConfiguration( TEST_NAME+i, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME+i, new String[] { String.valueOf(i) }) );
	}
	
	@Test
	public void testCodegenRowAggRewrite1CP() {
		testCodegenIntegration( TEST_NAME1, true, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg1CP() {
		testCodegenIntegration( TEST_NAME1, false, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg1SP() {
		testCodegenIntegration( TEST_NAME1, false, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenRowAggRewrite2CP() {
		testCodegenIntegration( TEST_NAME2, true, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg2CP() {
		testCodegenIntegration( TEST_NAME2, false, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg2SP() {
		testCodegenIntegration( TEST_NAME2, false, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenRowAggRewrite3CP() {
		testCodegenIntegration( TEST_NAME3, true, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg3CP() {
		testCodegenIntegration( TEST_NAME3, false, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg3SP() {
		testCodegenIntegration( TEST_NAME3, false, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenRowAggRewrite4CP() {
		testCodegenIntegration( TEST_NAME4, true, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg4CP() {
		testCodegenIntegration( TEST_NAME4, false, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg4SP() {
		testCodegenIntegration( TEST_NAME4, false, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenRowAggRewrite5CP() {
		testCodegenIntegration( TEST_NAME5, true, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg5CP() {
		testCodegenIntegration( TEST_NAME5, false, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg5SP() {
		testCodegenIntegration( TEST_NAME5, false, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenRowAggRewrite6CP() {
		testCodegenIntegration( TEST_NAME6, true, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg6CP() {
		testCodegenIntegration( TEST_NAME6, false, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg6SP() {
		testCodegenIntegration( TEST_NAME6, false, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenRowAggRewrite7CP() {
		testCodegenIntegration( TEST_NAME7, true, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg7CP() {
		testCodegenIntegration( TEST_NAME7, false, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg7SP() {
		testCodegenIntegration( TEST_NAME7, false, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenRowAggRewrite8CP() {
		testCodegenIntegration( TEST_NAME8, true, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg8CP() {
		testCodegenIntegration( TEST_NAME8, false, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg8SP() {
		testCodegenIntegration( TEST_NAME8, false, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenRowAggRewrite9CP() {
		testCodegenIntegration( TEST_NAME9, true, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg9CP() {
		testCodegenIntegration( TEST_NAME9, false, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg9SP() {
		testCodegenIntegration( TEST_NAME9, false, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenRowAggRewrite10CP() {
		testCodegenIntegration( TEST_NAME10, true, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg10CP() {
		testCodegenIntegration( TEST_NAME10, false, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg10SP() {
		testCodegenIntegration( TEST_NAME10, false, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenRowAggRewrite11CP() {
		testCodegenIntegration( TEST_NAME11, true, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg11CP() {
		testCodegenIntegration( TEST_NAME11, false, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg11SP() {
		testCodegenIntegration( TEST_NAME11, false, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenRowAggRewrite12CP() {
		testCodegenIntegration( TEST_NAME12, true, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg12CP() {
		testCodegenIntegration( TEST_NAME12, false, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg12SP() {
		testCodegenIntegration( TEST_NAME12, false, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenRowAggRewrite13CP() {
		testCodegenIntegration( TEST_NAME13, true, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg13CP() {
		testCodegenIntegration( TEST_NAME13, false, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg13SP() {
		testCodegenIntegration( TEST_NAME13, false, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenRowAggRewrite14CP() {
		testCodegenIntegration( TEST_NAME14, true, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg14CP() {
		testCodegenIntegration( TEST_NAME14, false, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg14SP() {
		testCodegenIntegration( TEST_NAME14, false, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenRowAggRewrite15CP() {
		testCodegenIntegration( TEST_NAME15, true, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg15CP() {
		testCodegenIntegration( TEST_NAME15, false, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg15SP() {
		testCodegenIntegration( TEST_NAME15, false, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenRowAggRewrite16CP() {
		testCodegenIntegration( TEST_NAME16, true, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg16CP() {
		testCodegenIntegration( TEST_NAME16, false, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg16SP() {
		testCodegenIntegration( TEST_NAME16, false, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenRowAggRewrite17CP() {
		testCodegenIntegration( TEST_NAME17, true, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg17CP() {
		testCodegenIntegration( TEST_NAME17, false, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg17SP() {
		testCodegenIntegration( TEST_NAME17, false, ExecType.SPARK );
	}
	
	@Test
	@Ignore
	// Since adding the rewrite (simplyfyMMCBindZeroVector) CodeGen is unable to
	// combine the instructions.
	public void testCodegenRowAggRewrite18CP() {
		testCodegenIntegration( TEST_NAME18, true, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg18CP() {
		testCodegenIntegration( TEST_NAME18, false, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg18SP() {
		testCodegenIntegration( TEST_NAME18, false, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenRowAggRewrite19CP() {
		testCodegenIntegration( TEST_NAME19, true, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg19CP() {
		testCodegenIntegration( TEST_NAME19, false, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg19SP() {
		testCodegenIntegration( TEST_NAME19, false, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenRowAggRewrite20CP() {
		testCodegenIntegration( TEST_NAME20, true, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg20CP() {
		testCodegenIntegration( TEST_NAME20, false, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg20SP() {
		testCodegenIntegration( TEST_NAME20, false, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenRowAggRewrite21CP() {
		testCodegenIntegration( TEST_NAME21, true, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg21CP() {
		testCodegenIntegration( TEST_NAME21, false, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg21SP() {
		testCodegenIntegration( TEST_NAME21, false, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenRowAggRewrite22CP() {
		testCodegenIntegration( TEST_NAME22, true, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg22CP() {
		testCodegenIntegration( TEST_NAME22, false, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg22SP() {
		testCodegenIntegration( TEST_NAME22, false, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenRowAggRewrite23CP() {
		testCodegenIntegration( TEST_NAME23, true, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg23CP() {
		testCodegenIntegration( TEST_NAME23, false, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg23SP() {
		testCodegenIntegration( TEST_NAME23, false, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenRowAggRewrite24CP() {
		testCodegenIntegration( TEST_NAME24, true, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg24CP() {
		testCodegenIntegration( TEST_NAME24, false, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg24SP() {
		testCodegenIntegration( TEST_NAME24, false, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenRowAggRewrite25CP() {
		testCodegenIntegration( TEST_NAME25, true, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg25CP() {
		testCodegenIntegration( TEST_NAME25, false, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg25SP() {
		testCodegenIntegration( TEST_NAME25, false, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenRowAggRewrite26CP() {
		testCodegenIntegration( TEST_NAME26, true, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg26CP() {
		testCodegenIntegration( TEST_NAME26, false, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg26SP() {
		testCodegenIntegration( TEST_NAME26, false, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenRowAggRewrite27CP() {
		testCodegenIntegration( TEST_NAME27, true, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg27CP() {
		testCodegenIntegration( TEST_NAME27, false, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg27SP() {
		testCodegenIntegration( TEST_NAME27, false, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenRowAggRewrite28CP() {
		testCodegenIntegration( TEST_NAME28, true, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg28CP() {
		testCodegenIntegration( TEST_NAME28, false, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg28SP() {
		testCodegenIntegration( TEST_NAME28, false, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenRowAggRewrite29CP() {
		testCodegenIntegration( TEST_NAME29, true, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg29CP() {
		testCodegenIntegration( TEST_NAME29, false, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg29SP() {
		testCodegenIntegration( TEST_NAME29, false, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenRowAggRewrite30CP() {
		testCodegenIntegration( TEST_NAME30, true, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg30CP() {
		testCodegenIntegration( TEST_NAME30, false, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg30SP() {
		testCodegenIntegration( TEST_NAME30, false, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenRowAggRewrite31CP() {
		testCodegenIntegration( TEST_NAME31, true, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg31CP() {
		testCodegenIntegration( TEST_NAME31, false, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg31SP() {
		testCodegenIntegration( TEST_NAME31, false, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenRowAggRewrite32CP() {
		testCodegenIntegration( TEST_NAME32, true, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg32CP() {
		testCodegenIntegration( TEST_NAME32, false, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg32SP() {
		testCodegenIntegration( TEST_NAME32, false, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenRowAggRewrite33CP() {
		testCodegenIntegration( TEST_NAME33, true, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg33CP() {
		testCodegenIntegration( TEST_NAME33, false, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg33SP() {
		testCodegenIntegration( TEST_NAME33, false, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenRowAggRewrite34CP() {
		testCodegenIntegration( TEST_NAME34, true, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg34CP() {
		testCodegenIntegration( TEST_NAME34, false, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg34SP() {
		testCodegenIntegration( TEST_NAME34, false, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenRowAggRewrite35CP() {
		testCodegenIntegration( TEST_NAME35, true, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg35CP() {
		testCodegenIntegration( TEST_NAME35, false, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg35SP() {
		testCodegenIntegration( TEST_NAME35, false, ExecType.SPARK );
	}

	@Test
	public void testCodegenRowAggRewrite36CP() {
		testCodegenIntegration( TEST_NAME36, true, ExecType.CP );
	}

	@Test
	public void testCodegenRowAgg36CP() {
		testCodegenIntegration( TEST_NAME36, false, ExecType.CP );
	}

	@Test
	public void testCodegenRowAgg36SP() {
		testCodegenIntegration( TEST_NAME36, false, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenRowAggRewrite37CP() {
		testCodegenIntegration( TEST_NAME37, true, ExecType.CP );
	}

	@Test
	public void testCodegenRowAgg37CP() {
		testCodegenIntegration( TEST_NAME37, false, ExecType.CP );
	}

	@Test
	public void testCodegenRowAgg37SP() {
		testCodegenIntegration( TEST_NAME37, false, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenRowAggRewrite38CP() {
		testCodegenIntegration( TEST_NAME38, true, ExecType.CP );
	}

	@Test
	public void testCodegenRowAgg38CP() {
		testCodegenIntegration( TEST_NAME38, false, ExecType.CP );
	}

	@Test
	public void testCodegenRowAgg38SP() {
		testCodegenIntegration( TEST_NAME38, false, ExecType.SPARK );
	}

	@Test
	public void testCodegenRowAggRewrite39CP() {
		testCodegenIntegration( TEST_NAME39, true, ExecType.CP );
	}

	@Test
	public void testCodegenRowAgg39CP() {
		testCodegenIntegration( TEST_NAME39, false, ExecType.CP );
	}

	@Test
	public void testCodegenRowAgg39SP() {
		testCodegenIntegration( TEST_NAME39, false, ExecType.SPARK );
	}

	@Test
	public void testCodegenRowAggRewrite40CP() {
		testCodegenIntegration( TEST_NAME40, true, ExecType.CP );
	}

	@Test
	public void testCodegenRowAgg40CP() {
		testCodegenIntegration( TEST_NAME40, false, ExecType.CP );
	}

	@Test
	public void testCodegenRowAgg40SP() {
		testCodegenIntegration( TEST_NAME40, false, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenRowAggRewrite41CP() {
		testCodegenIntegration( TEST_NAME41, true, ExecType.CP );
	}

	@Test
	public void testCodegenRowAgg41CP() {
		testCodegenIntegration( TEST_NAME41, false, ExecType.CP );
	}

	@Test
	public void testCodegenRowAgg41SP() {
		testCodegenIntegration( TEST_NAME41, false, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenRowAggRewrite42CP() {
		testCodegenIntegration( TEST_NAME42, true, ExecType.CP );
	}

	@Test
	public void testCodegenRowAgg42CP() {
		testCodegenIntegration( TEST_NAME42, false, ExecType.CP );
	}

	@Test
	public void testCodegenRowAgg42SP() {
		testCodegenIntegration( TEST_NAME42, false, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenRowAggRewrite43CP() {
		testCodegenIntegration( TEST_NAME43, true, ExecType.CP );
	}

	@Test
	public void testCodegenRowAgg43CP() {
		testCodegenIntegration( TEST_NAME43, false, ExecType.CP );
	}

	@Test
	public void testCodegenRowAgg43SP() {
		testCodegenIntegration( TEST_NAME43, false, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenRowAggRewrite44CP() {
		testCodegenIntegration( TEST_NAME44, true, ExecType.CP );
	}

	@Test
	public void testCodegenRowAgg44CP() {
		testCodegenIntegration( TEST_NAME44, false, ExecType.CP );
	}

	@Test
	public void testCodegenRowAgg44SP() {
		testCodegenIntegration( TEST_NAME44, false, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenRowAggRewrite45CP() {
		testCodegenIntegration( TEST_NAME45, true, ExecType.CP );
	}

	@Test
	public void testCodegenRowAgg45CP() {
		testCodegenIntegration( TEST_NAME45, false, ExecType.CP );
	}

	@Test
	public void testCodegenRowAgg45SP() {
		testCodegenIntegration( TEST_NAME45, false, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenRowAggRewrite46CP() {
		testCodegenIntegration( TEST_NAME46, true, ExecType.CP );
	}

	@Test
	public void testCodegenRowAgg46CP() {
		testCodegenIntegration( TEST_NAME46, false, ExecType.CP );
	}

	@Test
	public void testCodegenRowAgg46SP() {
		testCodegenIntegration( TEST_NAME46, false, ExecType.SPARK );
	}

	@Test
	public void testCodegenRowAggRewrite47CP() {
		testCodegenIntegration( TEST_NAME47, true, ExecType.CP );
	}

	@Test
	public void testCodegenRowAgg47CP() {
		testCodegenIntegration( TEST_NAME47, false, ExecType.CP );
	}

	@Test
	public void testCodegenRowAgg47SP() {
		testCodegenIntegration( TEST_NAME47, false, ExecType.SPARK );
	}

	@Test
	public void testCodegenRowAggRewrite48CP() {
		testCodegenIntegration( TEST_NAME48, true, ExecType.CP );
	}

	@Test
	public void testCodegenRowAgg48CP() {
		testCodegenIntegration( TEST_NAME48, false, ExecType.CP );
	}

	@Test
	public void testCodegenRowAgg48SP() {
		testCodegenIntegration( TEST_NAME48, false, ExecType.SPARK );
	}

	@Test
	public void testCodegenRowAgg49CP() {testCodegenIntegration( TEST_NAME49, false, ExecType.CP );}

	private void testCodegenIntegration( String testname, boolean rewrites, ExecType instType )
	{
		boolean oldFlag = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		ExecMode platformOld = setExecMode(instType);
		
		try
		{
			TestConfiguration config = getTestConfiguration(testname);
			loadTestConfiguration(config);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[]{"-explain", "codegen", "-sparseIntermediate", "-stats", "-args", output("S") };
			
			fullRScriptName = HOME + testname + ".R";
			rCmd = getRCmd(inputDir(), expectedDir());

			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;
			
			runTest(true, false, null, -1);
			runRScript(true);
			
			if(testname.equals(TEST_NAME38) && TEST_GPU)
				eps = Math.pow(10, -7);
			//compare matrices
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("S");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromExpectedDir("S");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
			Assert.assertTrue(heavyHittersContainsSubString("spoofRA") 
				|| heavyHittersContainsSubString("sp_spoofRA") 
				|| heavyHittersContainsSubString("gpu_spoofCUDARA"));
			
			//ensure full aggregates for certain patterns
			if( testname.equals(TEST_NAME15) )
				Assert.assertFalse(heavyHittersContainsSubString(Opcodes.UARKP.toString()));
			if( testname.equals(TEST_NAME17) )
				Assert.assertFalse(heavyHittersContainsSubString(Opcodes.RIGHT_INDEX.toString()));
			if( testname.equals(TEST_NAME28) || testname.equals(TEST_NAME45) )
				Assert.assertTrue(!heavyHittersContainsSubString("spoofRA", 2)
					&& !heavyHittersContainsSubString("sp_spoofRA", 2));
			if( testname.equals(TEST_NAME30) )
				Assert.assertTrue(!heavyHittersContainsSubString("spoofRA", 2)
					&& !heavyHittersContainsSubString(Opcodes.RIGHT_INDEX.toString()));
			if( testname.equals(TEST_NAME31) )
				Assert.assertFalse(heavyHittersContainsSubString("spoofRA", 2));
			if( testname.equals(TEST_NAME35) )
				Assert.assertTrue(!heavyHittersContainsSubString("spoofRA", 2)
					&& !heavyHittersContainsSubString(Opcodes.CBIND.toString()));
			if( testname.equals(TEST_NAME36) )
				Assert.assertFalse(heavyHittersContainsSubString(Opcodes.XOR.toString()));
			if( testname.equals(TEST_NAME41) )
				Assert.assertFalse(heavyHittersContainsSubString("seq"));
			if( testname.equals(TEST_NAME42) )
				Assert.assertTrue(!heavyHittersContainsSubString(Opcodes.MIN.toString(),Opcodes.NMIN.toString())
					&& !heavyHittersContainsSubString(Opcodes.SPOOF.toString(), 2));
			if( testname.equals(TEST_NAME44) )
				Assert.assertTrue(!heavyHittersContainsSubString(Opcodes.MAXPOOLING.toString())
					&& !heavyHittersContainsSubString(Opcodes.SPOOF.toString(), 2));
			if( testname.equals(TEST_NAME46) )
				Assert.assertTrue(!heavyHittersContainsSubString(Opcodes.CONV2D.toString())
					&& !heavyHittersContainsSubString(Opcodes.SPOOF.toString(), 2));
		}
		finally {
			resetExecMode(platformOld);
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldFlag;
			OptimizerUtils.ALLOW_AUTO_VECTORIZATION = true;
			OptimizerUtils.ALLOW_OPERATOR_FUSION = true;
		}
	}

	/**
	 * Override default configuration with custom test configuration to ensure
	 * scratch space and local temporary directory locations are also updated.
	 */
	@Override
	protected File getConfigTemplateFile() {
		// Instrumentation in this test's output log to show custom configuration file used for template.
		LOG.debug("This test case overrides default configuration with " + TEST_CONF_FILE.getPath());
		return TEST_CONF_FILE;
	}
}
