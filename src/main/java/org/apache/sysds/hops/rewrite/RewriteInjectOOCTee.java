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

package org.apache.sysds.hops.rewrite;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.TeeOp;

import java.util.ArrayList;

public class RewriteInjectOOCTee extends HopRewriteRule {

    /**
     * Handle a generic (last-level) hop DAG with multiple roots.
     *
     * @param roots high-level operator roots
     * @param state program rewrite status
     * @return list of high-level operators
     */
    @Override
    public ArrayList<Hop> rewriteHopDAGs(ArrayList<Hop> roots, ProgramRewriteStatus state) {
        if (roots == null) {
            return null;
        }
        for  (Hop root : roots) {
            rewriteHopDAG(root, state);
        }
        return roots;
    }

    /**
     * Handle a predicate hop DAG with exactly one root.
     *
     * @param root  high-level operator root
     * @param state program rewrite status
     * @return high-level operator
     */
    @Override
    public Hop rewriteHopDAG(Hop root, ProgramRewriteStatus state) {
        if (root.isVisited()) {
            return root;
        }

        for (int i = 0; i < root.getInput().size(); i++) {
            root.getInput().set(i, rewriteHopDAG(root.getInput().get(i), state));
        }

        applyTeeRewrite(root, new ArrayList<Hop>(root.getParent()));

        root.setVisited(true);
        return root;
    }

    /**
     *
     *
     * @param hop
     * @param parents
     */
    private void applyTeeRewrite(Hop hop, ArrayList<Hop> parents) {
        // --- RULE TRIGGERS ---
        // 1. Is the operation an OOC operation?
        // 2. Does it have more than one parent (i.e., is it consumed by multiple operations)?
        // 3. Is it not already a Tee operation (to prevent infinite rewrite loops)?
        boolean isOOC = (hop.getForcedExecType() == Types.ExecType.OOC);
        boolean multipleConsumers = parents.size() > 1;
        boolean isNotAlreadyTee =  !(hop instanceof TeeOp);
        boolean isOOCEnabled = DMLScript.USE_OOC;


        if ( isOOCEnabled && multipleConsumers && isNotAlreadyTee) {
            System.out.println("perform rewrite");
        }

    }
}
