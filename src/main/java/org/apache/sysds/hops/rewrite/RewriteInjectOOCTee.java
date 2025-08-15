package org.apache.sysds.hops.rewrite;

import org.apache.sysds.hops.Hop;

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

        root.setVisited(true);
        return root;
    }
}
