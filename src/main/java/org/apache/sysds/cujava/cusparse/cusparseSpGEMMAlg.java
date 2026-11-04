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

package org.apache.sysds.cujava.cusparse;

public class cusparseSpGEMMAlg {

	public static final int CUSPARSE_SPGEMM_DEFAULT = 0;

	public static final int CUSPARSE_SPGEMM_CSR_ALG_DETERMINITIC = 1;

	public static final int CUSPARSE_SPGEMM_CSR_ALG_NONDETERMINITIC = 2;

	public static final int CUSPARSE_SPGEMM_ALG1 = 3;

	public static final int CUSPARSE_SPGEMM_ALG2 = 4;

	public static final int CUSPARSE_SPGEMM_ALG3 = 5;

	private cusparseSpGEMMAlg() {
		// Private constructor to prevent instantiation
	}
}
