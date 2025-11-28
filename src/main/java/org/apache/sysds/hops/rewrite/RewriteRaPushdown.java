package org.apache.sysds.hops.rewrite;

import org.apache.sysds.common.Types;
import org.apache.sysds.hops.*;
import org.apache.sysds.hops.recompile.Recompiler;
import org.apache.sysds.parser.StatementBlock;
import org.apache.sysds.parser.VariableSet;

import java.util.ArrayList;
import java.util.List;

/**
 * Rule: Simplify program structure by rewriting relational expressions,
 * implemented here: Pushdown of Selections before Join.
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
        ret.add(sb);
        return ret;
    }

    @Override
    public List<StatementBlock> rewriteStatementBlocks(List<StatementBlock> sbs, ProgramRewriteStatus state) {
        if (sbs == null || sbs.size() <= 1)
            return sbs;

        ArrayList<StatementBlock> tmpList = new ArrayList<>(sbs);
        boolean changed = false;

        // iterate over all SBs including a FuncOp with FuncName m_raJoin
        for (int i : findFunctionSb(tmpList, "m_raJoin", 0)){
            StatementBlock sb1 = tmpList.get(i);
            FunctionOp joinOp = findFunctionOp(sb1.getHops(),  "m_raJoin");

            // iterate over all following SBs including a FuncOp with FuncName m_raSelection
            for (int j : findFunctionSb(tmpList, "m_raSelection", i+1)){
                StatementBlock sb2 = tmpList.get(j);
                FunctionOp selOp = findFunctionOp(sb2.getHops(), "m_raSelection");

                // create deep copy to ensure data consistency
                FunctionOp tmpJoinOp = (FunctionOp) Recompiler.deepCopyHopsDag(joinOp);
                FunctionOp tmpSelOp = (FunctionOp) Recompiler.deepCopyHopsDag(selOp);

                if (!checkDataDependency(tmpJoinOp, tmpSelOp)){continue;}

                Hop selColHop = tmpSelOp.getInput(1);
                long selCol = getConstantSelectionCol(selColHop);
                if (selCol <= 0)
                    continue;

                // collect Variable Sets
                VariableSet joinRead = new VariableSet(sb1.variablesRead());
                VariableSet joinUpdated = new VariableSet(sb1.variablesUpdated());
                VariableSet selRead = new VariableSet(sb2.variablesRead());

                // join inputs: [A, colA, B, colB, method]
                long colsLeft = tmpJoinOp.getInput(0).getDataCharacteristics().getCols();
                long colsRight = tmpJoinOp.getInput(2).getDataCharacteristics().getCols();
                if (colsLeft <= 0 || colsRight <= 0)
                    continue;

                // decide which side of inner join the selection belongs to (A / B)
                int selSideIdx;
                if (selCol <= colsLeft) {
                    selSideIdx = 0;
                }
                else if (selCol <= colsLeft + colsRight) {
                    selSideIdx = 2;
                    LiteralOp adjustedColHop = new LiteralOp(selCol - colsLeft);
                    adjustedColHop.setName(selColHop.getName());
                    HopRewriteUtils.replaceChildReference(tmpSelOp, selColHop, adjustedColHop, 1);
                }
                else { continue; } // invalid column index

                // switch funcOps Output Variables
                String joinOutVar = tmpJoinOp.getOutputVariableNames()[0];
                tmpJoinOp.getOutputVariableNames()[0] = tmpSelOp.getOutputVariableNames()[0];
                tmpSelOp.getOutputVariableNames()[0] = joinOutVar;

                // rewire selection to consume the correct join input and adjusted column
                Hop newSelInput = tmpJoinOp.getInput().get(selSideIdx);
                HopRewriteUtils.replaceChildReference(tmpSelOp, tmpSelOp.getInput().get(0), newSelInput, 0);

                // let the join take selection output instead of raw input
                Hop newJoinInput = HopRewriteUtils.createTransientRead(joinOutVar, tmpSelOp);
                HopRewriteUtils.replaceChildReference(tmpJoinOp, newSelInput, newJoinInput, selSideIdx);

                //switch StatementBlock-assignments
                sb1.getHops().remove(joinOp);
                sb1.getHops().add(tmpSelOp);
                sb2.getHops().remove(selOp);
                sb2.getHops().add(tmpJoinOp);

                // modify SB- variable sets
                VariableSet vs = new VariableSet();
                vs.addVariable(joinOutVar, joinUpdated.getVariable(joinOutVar));
                selRead.removeVariables(vs);
                selRead.addVariable(newSelInput.getName(), joinRead.getVariable(newSelInput.getName()));

                // selection now reads the original join inputs plus its own metadata
                sb1.setReadVariables(selRead);
                sb1.setLiveOut(VariableSet.minus(joinUpdated, selRead));
                sb1.setLiveIn(selRead);
                sb1.setGen(selRead);

                // join now consumes the selection output and produces the output
                sb2.setReadVariables(sb1.liveOut());
                sb2.setGen(sb1.liveOut());
                sb2.setLiveIn(sb1.liveOut());

                // mark change & increment i by 1 (i+1 = now join-Sb)
                changed = true;
                i++;

                LOG.debug("Applied rewrite: pushed m_raSelection before m_raJoin (blocks lines "
                        + sb1.getBeginLine() + "-" + sb1.getEndLine() + " and "
                        + sb2.getBeginLine() + "-" + sb2.getEndLine() + ").");
            }
        }
        return changed ? tmpList : sbs;
    }

    private List<Integer> findFunctionSb(List<StatementBlock> sbs, String functionName, int startIdx) {
        List<Integer> functionSbs = new ArrayList<>();

        for (int i = startIdx; i < sbs.size(); i++) {
            StatementBlock sb = sbs.get(i);

            // easy preconditions
            if (!HopRewriteUtils.isLastLevelStatementBlock(sb) || sb.isSplitDag()) {
                continue;
            }

            // find if StatementBlocks have certain FunctionOp, continue if not found
            FunctionOp functionOp = findFunctionOp(sb.getHops(), functionName);

            // if found, add to list
            if (functionOp != null) { functionSbs.add(i); }
        }

        return functionSbs;
    }

    private boolean checkDataDependency(FunctionOp fOut, FunctionOp fIn){
        for (String out : fOut.getOutputVariableNames()) {
            for (Hop h : fIn.getInput()) {
                if (h.getName().equals(out)){
                    return true;
                }
            }
        }
        return false;
    }

    private FunctionOp findFunctionOp(List<Hop> roots, String functionName) {
        if (roots == null)
            return null;
        Hop.resetVisitStatus(roots, true);
        for (Hop root : roots) {
            if (root instanceof FunctionOp funcOp) {
                if (funcOp.getFunctionName().equals(functionName))
                { return funcOp; }
            }
        }
        return null;
    }

    private long getConstantSelectionCol(Hop selColHop) {
        if (selColHop instanceof LiteralOp lit)
            return HopRewriteUtils.getIntValueSafe(lit);

        // Handle casted literals (e.g., type propagation inserted casts)
        if (selColHop instanceof UnaryOp uop && uop.getOp() == Types.OpOp1.CAST_AS_INT
                && uop.getInput().get(0) instanceof LiteralOp lit)
            return HopRewriteUtils.getIntValueSafe(lit);

        // If hop is a dataop whose input is a literal, try to fold
        if (selColHop instanceof DataOp dop && !dop.getInput().isEmpty() && dop.getInput().get(0) instanceof LiteralOp lit)
            return HopRewriteUtils.getIntValueSafe(lit);

        return -1; // unknown at rewrite time
    }
}