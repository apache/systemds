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

import java.util.ArrayList;

public class EOpNodeData extends EOpNode {
    public int matrixIdx;
    public EOpNodeData(Character c1, Character c2, int matrixIdx){
        super(c1,c2);
        this.matrixIdx = matrixIdx;
    }
	@Override
	public String[] recursivePrintString() {
		String[] res = new String[1];
		res[0] = this.getClass().getSimpleName()+" ("+matrixIdx+") "+this.toString();
		return res;
	}
    @Override
    public MatrixBlock computeEOpNode(ArrayList<MatrixBlock> inputs, int numOfThreads, Log LOG) {
        return inputs.get(matrixIdx);
    }

    @Override
    public void reorderChildren(Character outChar1, Character outChar2) {

    }
}
