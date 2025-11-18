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

package org.apache.sysds.runtime.einsum;

import org.apache.commons.logging.Log;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import scala.Int;

import java.util.ArrayList;

public abstract class EOpNode {
    public Character c1;
    public Character c2;
	public Integer dim1;
	public Integer dim2;
    public EOpNode(Character c1, Character c2, Integer dim1, Integer dim2) {
        this.c1 = c1;
        this.c2 = c2;
		this.dim1 = dim1;
		this.dim2 = dim2;
    }

    @Override
    public String toString() {
        if(c1 == null) return "''";
        if(c2 == null) return c1.toString();
        return c1.toString() + c2.toString();
    }

	public abstract String[] recursivePrintString();

    public abstract MatrixBlock computeEOpNode(ArrayList<MatrixBlock> inputs, int numOfThreads, Log LOG);

    public abstract EOpNode reorderChildrenAndOptimize(EOpNode parent, Character outChar1, Character outChar2);
}

