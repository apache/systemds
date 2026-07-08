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

package org.apache.sysds.test.applications.nn;

import org.junit.Test;

public class NNOptimTest extends TestFolder {

	@Test
	public void sgd() {
		run("sgd.dml");
	}

	@Test
	public void sgd_momentum() {
		run("sgd_momentum.dml");
	}

	@Test
	public void rmsprop() {
		run("rmsprop.dml");
	}

	@Test
	public void sgd_nesterov() {
		run("sgd_nesterov.dml");
	}

	@Test
	public void adagrad() {
		run("adagrad.dml");
	}

	@Test
	public void adam() {
		run("adam.dml");
	}

	@Test
	public void adamw() {
		run("adamw.dml");
	}

	@Test
	public void lars() {
		run("lars.dml");
	}

	@Override
	protected void run(String name) {
		super.run("component/optim/" + name);
	}
}
