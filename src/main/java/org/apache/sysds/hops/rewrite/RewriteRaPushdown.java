package org.apache.sysds.hops.rewrite;

import org.apache.sysds.parser.StatementBlock;

import java.util.ArrayList;
import java.util.List;

/**
 * Rule: Simplify program structure by rewriting relational expressions,
 * i.e. Pushdown of Selections
 */
public class RewriteRaPushdown extends StatementBlockRewriteRule
{

    @Override
    public boolean createsSplitDag() {
        return false;
    }

    @Override
    public List<StatementBlock> rewriteStatementBlock(StatementBlock sb, ProgramRewriteStatus state) {
        ArrayList<StatementBlock> ret = new ArrayList<>();
        boolean apply = false;

        /*
         * #1 Split n-ary joins into binary joins
         * #2 Split multi-term selections
         * #3 Push-down selections as far as possible
         * #4 Group adjacent selections again
         * #5 Push-down projections as far as possible
         */

        LOG.debug("Applied rewriteRaPushdown (lines "+sb.getBeginLine()+"-"+sb.getEndLine()+").");
        ret.add(sb);

        return ret;
    }

    @Override
    public List<StatementBlock> rewriteStatementBlocks(List<StatementBlock> sbs, ProgramRewriteStatus state) {
        return sbs;
    }
}
