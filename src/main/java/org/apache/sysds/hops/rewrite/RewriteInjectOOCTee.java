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
import org.apache.sysds.hops.MemoTable;
import org.apache.sysds.hops.TeeOp;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.runtime.meta.DataCharacteristics;

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

            // Take a defensive copy even before any rewrite
            ArrayList<Hop> consumers = new ArrayList<>(hop.getParent());

            // 1. Create a list of placeholder hops for tee outputs
//            ArrayList<Hop> teeOutputs = new ArrayList<>();
//            for (int i = 0 ; i < parents.size() ; i++) {
//                teeOutputs.add(new Hop("tee_out_"+i, hop.getDataType(), hop.getValueType()) {
//                                   @Override
//                                   public boolean allowsAllExecTypes() {
//                                       return false;
//                                   }
//
//                                   @Override
//                                   protected DataCharacteristics inferOutputCharacteristics(MemoTable memo) {
//                                       return null;
//                                   }
//
//                                   @Override
//                                   public Lop constructLops() {
//                                       if (this.getLops() == null) {
//                                           System.out.println("we are at constructLops");
//                                           this.setLops(hop.getLops());
//                                       }
//                                       return this.getLops();
//                                   }
//
//                                   @Override
//                                   protected Types.ExecType optFindExecType(boolean transitive) {
//                                       return null;
//                                   }
//
//                                   @Override
//                                   public String getOpString() {
//                                       return "";
//                                   }
//
//                                   @Override
//                                   public boolean isGPUEnabled() {
//                                       return false;
//                                   }
//
//                                   @Override
//                                   protected double computeOutputMemEstimate(long dim1, long dim2, long nnz) {
//                                       return 0;
//                                   }
//
//                                   @Override
//                                   protected double computeIntermediateMemEstimate(long dim1, long dim2, long nnz) {
//                                       return 0;
//                                   }
//
//                                   @Override
//                                   public void refreshSizeInformation() {
//
//                                   }
//
//                                   @Override
//                                   public Object clone() throws CloneNotSupportedException {
//                                       return null;
//                                   }
//
//                                   @Override
//                                   public boolean compare(Hop that) {
//                                       return false;
//                                   }
//                               }
//                );
//            }

            // 2. Create the new TeeOp. Take original hop as input
            TeeOp teeOp = new TeeOp(hop);

            // 3. Rewire the graph:
            //   For each original consumer, change its input from the original hop
            //   to one of the new outputs of the TeeOp
//            ArrayList<Hop> consumers = new ArrayList<>(hop.getParent());
            for (int i = 0 ; i < consumers.size() ; i++) {
                Hop consumer = consumers.get(i);
//                Hop teeOuput = teeOp.getOutput(i);
                HopRewriteUtils.replaceChildReference(consumer, hop, teeOp);
            }

        }

    }
}
