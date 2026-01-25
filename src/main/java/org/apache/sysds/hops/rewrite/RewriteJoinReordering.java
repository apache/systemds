package org.apache.sysds.hops.rewrite;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.hops.DataOp;
import org.apache.sysds.hops.FunctionOp;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.LiteralOp;
import org.apache.sysds.parser.StatementBlock;
import org.apache.sysds.parser.VariableSet;
import org.apache.sysds.parser.WhileStatement;
import org.apache.sysds.parser.WhileStatementBlock;
import org.apache.sysds.parser.DataIdentifier;
import org.apache.sysds.parser.ForStatement;
import org.apache.sysds.parser.ForStatementBlock;
import org.apache.sysds.parser.FunctionStatement;
import org.apache.sysds.parser.FunctionStatementBlock;
import org.apache.sysds.parser.IfStatement;
import org.apache.sysds.parser.IfStatementBlock;

public class RewriteJoinReordering extends StatementBlockRewriteRule {
  // This exception is thrown when we cannot determine the base dependencies of a
  // given join.
  private class UnknownCanonicalJoinException extends RuntimeException {
    private UnknownCanonicalJoinException() {
      super();
    }
  }

  // This exception is thrown when we cannot determine the dimension information
  // for a given non-raJoin HOP.
  private class UnknownDimensionInfoException extends RuntimeException {
    private UnknownDimensionInfoException() {
      super();
    }
  }

  private boolean isRaJoin(Hop node) {
    if (node instanceof FunctionOp fnode) {
      return fnode.getFunctionNamespace().equals(".builtinNS")
          && fnode.getFunctionName().equals("m_raJoin");
    }
    return false;
  }

  private boolean isLiteralInt(Hop node) {
    if (node instanceof LiteralOp) {
      return node.getValueType() == ValueType.INT64;
    }
    return false;
  }

  private boolean isKnownMatrix(Hop hop) {
    return hop.getDim1() > 0 && hop.getDim2() > 0;
  }

  @Override
  public boolean createsSplitDag() {
    return false;
  }

  /**
   * Collect all raJoin calls
   * 
   * @param sb      current statement block to search from
   * @param joinMap a mapping from the bound output variable name to the index of
   *                the join in the `joins` list.
   * @param joins   a list to accumulate all found raJoins
   */
  private void collectRaJoin(HashMap<Hop, StatementBlock> hopToSb, StatementBlock sb, HashMap<String, Integer> joinMap,
      ArrayList<FunctionOp> joins) {
    if (sb instanceof FunctionStatementBlock) {
      FunctionStatementBlock fsb = (FunctionStatementBlock) sb;
      FunctionStatement fstmt = (FunctionStatement) fsb.getStatement(0);
      for (StatementBlock sbi : fstmt.getBody())
        collectRaJoin(hopToSb, sbi, joinMap, joins);
    } else if (sb instanceof WhileStatementBlock) {
      WhileStatementBlock wsb = (WhileStatementBlock) sb;
      WhileStatement wstmt = (WhileStatement) wsb.getStatement(0);
      for (StatementBlock sbi : wstmt.getBody())
        collectRaJoin(hopToSb, sbi, joinMap, joins);
    } else if (sb instanceof IfStatementBlock) {
      IfStatementBlock isb = (IfStatementBlock) sb;
      IfStatement istmt = (IfStatement) isb.getStatement(0);
      for (StatementBlock sbi : istmt.getIfBody())
        collectRaJoin(hopToSb, sbi, joinMap, joins);
      for (StatementBlock sbi : istmt.getElseBody())
        collectRaJoin(hopToSb, sbi, joinMap, joins);
    } else if (sb instanceof ForStatementBlock) // incl parfor
    {
      ForStatementBlock fsb = (ForStatementBlock) sb;
      ForStatement fstmt = (ForStatement) fsb.getStatement(0);
      for (StatementBlock sbi : fstmt.getBody())
        collectRaJoin(hopToSb, sbi, joinMap, joins);
    } else // generic (last-level)
    {
      /*
       * Check for raJoins at this branch
       */
      for (Hop hop : sb.getHops()) {
        if (isRaJoin(hop)) {
          FunctionOp fhop = (FunctionOp) hop;
          processRaJoin(sb, hopToSb, fhop, joinMap, joins);
        }
      }
    }
  }

  /**
   * Add an raJoin HOP to custom intermediate objects.
   * 
   * @param fhop    the raJoin Hop
   * @param joinMap a mapping from the bound output variable name to the index of
   *                the join in the `joins` list.
   * @param joins   a list to accumulate all found raJoins
   */
  private void processRaJoin(StatementBlock sb, HashMap<Hop, StatementBlock> hopToSb, FunctionOp fhop,
      HashMap<String, Integer> joinMap, ArrayList<FunctionOp> joins) {
    Hop acol = fhop.getInput(1);
    Hop bcol = fhop.getInput(3);
    // only support literal values.
    if (!isLiteralInt(acol) || !isLiteralInt(bcol)) {
      return;
    }

    for (String varName : fhop.getOutputVariableNames()) {
      joinMap.put(varName, joins.size());
    }
    joins.add(fhop);
    hopToSb.put(fhop, sb);
  }

  /**
   * Find the topological order of all joins.
   * 
   * @param joinMap
   * @param joins
   * @return the topological order of joins as indices of `joins`
   */
  private ArrayList<Integer> topoOrder(HashMap<String, Integer> joinMap, ArrayList<FunctionOp> joins) {
    ArrayList<Integer> topoOrder = new ArrayList<>();
    boolean[] visited = new boolean[joins.size()];
    for (int i = 0; i < joins.size(); i++)
      dfsOrder(joinMap, joins, topoOrder, visited, i);
    Collections.reverse(topoOrder);
    return topoOrder;
  }

  /**
   * DFS call to find the topological order.
   * 
   * @param joinMap
   * @param joins
   * @param order
   * @param visited
   * @param i       the current join index we are at
   */
  private void dfsOrder(HashMap<String, Integer> joinMap, ArrayList<FunctionOp> joins, ArrayList<Integer> order,
      boolean[] visited, int i) {
    visited[i] = true;
    FunctionOp join = joins.get(i);
    Hop a = join.getInput(0);
    Hop b = join.getInput(2);
    // recurse if the matrix is not a base matrix.
    if (!isKnownMatrix(a)) {
      Integer next = joinMap.get(a.getName());
      if (next == null)
        throw new UnknownCanonicalJoinException();
      if (!visited[next]) {
        dfsOrder(joinMap, joins, order, visited, next);
      }
    }
    if (!isKnownMatrix(b)) {
      Integer next = joinMap.get(b.getName());
      if (next == null)
        throw new UnknownCanonicalJoinException();
      if (!visited[next]) {
        dfsOrder(joinMap, joins, order, visited, next);
      }
    }
    order.add(i);
  }

  /**
   * rewrite all roots
   * 
   * @param joinMap
   * @param joins   all raJoins
   * @param order   topological order of joins
   */
  private void rewriteRoots(ArrayList<StatementBlock> sbs, HashMap<Hop, StatementBlock> hopToSb,
      HashMap<String, Integer> joinMap, ArrayList<FunctionOp> joins, ArrayList<Integer> order) {
    boolean[] visited = new boolean[joins.size()];
    for (int i : order) {
      if (!visited[i]) {
        try {
          rewriteRoot(sbs, hopToSb, joinMap, joins, visited, i);
        } catch (Exception e) {
          // if it is a local exception, try rewriting the next root.
          if ((e instanceof UnknownCanonicalJoinException) || (e instanceof UnknownDimensionInfoException)) {
            continue;
          }
          throw e;
        }
      }
    }

    HashSet<Hop> consumedHops = new HashSet<>();
    for (int i = 0; i < joins.size(); i++) {
      if (!visited[i])
        continue;
      consumedHops.add(joins.get(i));
      HopRewriteUtils.cleanupUnreferenced(joins.get(i));
    }
    for (Hop hop : hopToSb.keySet()) {
      if (!consumedHops.contains(hop))
        continue;
      hopToSb.get(hop).getHops().remove(hop);
    }
  }

  // Custom representation of nested join calls.
  sealed interface JoinNode permits BaseNode, BinaryNode {
  }

  private record BaseNode(int i) implements JoinNode {
  };

  private record BinaryNode(JoinNode left, long leftCol, JoinNode right, long rightCol, String method)
      implements JoinNode {
  };

  private record Cost(long dim1, long dim2, long cost, JoinNode node) {
  };

  // Rewrite a single root
  private void rewriteRoot(ArrayList<StatementBlock> sbs, HashMap<Hop, StatementBlock> hopToSb,
      HashMap<String, Integer> joinMap, ArrayList<FunctionOp> joins, boolean[] visited, int rootIndex) {
    // get bases traversal = base relations(matrices)
    FunctionOp root = joins.get(rootIndex);
    ArrayList<Hop> bases = new ArrayList<>();
    ArrayList<Long> basesLengthPrefixSum = new ArrayList<>();
    ArrayList<CanonicalJoin> canonicalJoins = new ArrayList<>();
    dfsInorder(joinMap, joins, canonicalJoins, visited, bases, basesLengthPrefixSum, rootIndex);
    // convert all joins to joins between base relations.
    HashMap<BitSet, Cost> dp = new HashMap<>();
    for (int i = 0; i < bases.size() - 1; i++) {
      BitSet leftBS = new BitSet();
      BitSet rightBS = new BitSet();
      leftBS.set(i);
      rightBS.set(i + 1);
      CanonicalJoin validJoin = getValidJoin(canonicalJoins, leftBS, rightBS);
      if (validJoin == null) {
        continue;
      }
      BitSet bs = new BitSet(bases.size());
      bs.set(i);
      bs.set(i + 1);
      Hop left = bases.get(i);
      Hop right = bases.get(i + 1);

      long dim1 = left.getDim1() * right.getDim1();
      long dim2 = left.getDim2() + right.getDim2();
      long cost = dim1 * dim2;

      long leftCol = validJoin.acol;
      long rightCol = validJoin.bcol;

      JoinNode joinNode = new BinaryNode(new BaseNode(i), leftCol, new BaseNode(i + 1), rightCol, validJoin.method);
      dp.put(bs, new Cost(dim1, dim2, cost, joinNode));
    }
    for (int intervalLength = 2; intervalLength < bases.size(); intervalLength++) {
      // join base relation from the left
      for (int start = 1; start + intervalLength <= bases.size(); start++) {
        BitSet leftBS = new BitSet(bases.size());
        leftBS.set(start - 1);
        BitSet rightBS = new BitSet(bases.size());
        rightBS.set(start, start + intervalLength);
        if (dp.get(rightBS) == null) {
          continue;
        }
        CanonicalJoin validJoin = getValidJoin(canonicalJoins, leftBS, rightBS);
        if (validJoin == null) {
          continue;
        }

        BitSet bs = new BitSet(bases.size());
        bs.set(start - 1, start + intervalLength);

        Hop left = bases.get(start - 1);

        Cost right = dp.get(rightBS);

        long dim1 = left.getDim1() * right.dim1;
        long dim2 = left.getDim2() + right.dim2;
        long cost = dim1 * dim2 + right.cost;

        long leftCol = validJoin.acol;
        long rightCol = getRelativeCol(basesLengthPrefixSum, start, validJoin.bBaseIndex, validJoin.bcol);
        JoinNode joinNode = new BinaryNode(new BaseNode(start - 1), leftCol, right.node, rightCol, validJoin.method);
        dp.put(bs, new Cost(dim1, dim2, cost, joinNode));
      }
      // join base relation from the right
      for (int start = 0; start + intervalLength + 1 <= bases.size(); start++) {
        BitSet leftBS = new BitSet(bases.size());
        leftBS.set(start, start + intervalLength);
        BitSet rightBS = new BitSet(bases.size());
        rightBS.set(start + intervalLength);
        BitSet bs = new BitSet(bases.size());
        bs.set(start, start + intervalLength + 1);

        if (dp.get(leftBS) == null)
          continue;
        CanonicalJoin validJoin = getValidJoin(canonicalJoins, leftBS, rightBS);
        if (validJoin == null)
          continue;
        BitSet leftBs = new BitSet(bases.size());
        leftBs.set(start, start + intervalLength);
        Cost left = dp.get(leftBs);

        Hop right = bases.get(start + intervalLength);

        long dim1 = left.dim1 * right.getDim1();
        long dim2 = left.dim2 + right.getDim2();
        long cost = dim1 * dim2 + left.cost;

        if (dp.get(bs) == null || cost < dp.get(bs).cost) {
          long leftCol = getRelativeCol(basesLengthPrefixSum, start, validJoin.aBaseIndex, validJoin.acol);
          long rightCol = validJoin.bcol;
          JoinNode joinNode = new BinaryNode(left.node(), leftCol, new BaseNode(start + intervalLength), rightCol,
              validJoin.method);
          dp.put(bs, new Cost(dim1, dim2, cost, joinNode));
        }
      }
    }
    BitSet fullBs = new BitSet(bases.size());
    fullBs.set(0, bases.size());
    JoinNode optimalJoin = dp.get(fullBs).node;
    // System.out.println("optimalJoin: " + optimalJoin);

    // rewire the nodes.
    StatementBlock rootSb = hopToSb.get(root);
    ArrayList<Hop> rootSbHops = hopToSb.get(root).getHops();

    ArrayList<DataOp> intermediateWrites = new ArrayList<>();
    Hop newHop = generateHop(root, intermediateWrites, bases, optimalJoin);

    // remove and replace root
    for (int i = 0; i < rootSbHops.size(); i++) {
      if (rootSbHops.get(i) == root) {
        rootSbHops.set(i, newHop);
      }
    }
    HopRewriteUtils.rewireAllParentChildReferences(root, newHop);

    // remove all consumed joins that now aren't used
    HashSet<Hop> consumed = new HashSet<>();
    for (int j = 0; j < joins.size(); j++)
      if (visited[j])
        consumed.add(joins.get(j));

    rootSbHops.removeIf(consumed::contains);

    // rootSbHops.addAll(0,intermediateWrites);
    // add new Sb containing TWrites to right before it is consumed
    StatementBlock newSb = createIntermediateStatementBlock(rootSb, intermediateWrites);
    sbs.add(sbs.indexOf(rootSb), newSb);
  }

  // get the column number relative to the current relation starting at
  // `intervalStart
  long getRelativeCol(ArrayList<Long> prefixSum, int intervalStart, int baseIndex, long col) {
    long offset = col;
    if (intervalStart - 1 >= 0)
      offset -= prefixSum.get(intervalStart - 1);
    if (baseIndex - 1 >= 0)
      offset += prefixSum.get(baseIndex - 1);
    return offset;
  }

  // modified from RewriteHoistLoopInvariantOperations.java
  private StatementBlock createIntermediateStatementBlock(StatementBlock originalSb, List<DataOp> intermediateWrites) {
    //create empty last-level statement block
    StatementBlock ret = new StatementBlock();
    ret.setDMLProg(originalSb.getDMLProg());
    ret.setParseInfo(originalSb);
    ret.setLiveIn(new VariableSet(originalSb.liveIn()));
    ret.setLiveOut(new VariableSet(originalSb.liveIn()));

    //put custom hops
    ret.setHops(new ArrayList<>(intermediateWrites));

    // live variable analysis
    for (DataOp tWrite : intermediateWrites) {
      String varName = tWrite.getName();
      Hop hop = tWrite.getInput().get(0);
      DataIdentifier diVar = new DataIdentifier(varName);
      diVar.setDimensions(hop.getDim1(), hop.getDim2());
      diVar.setBlocksize(hop.getBlocksize());
      diVar.setDataType(hop.getDataType());
      diVar.setValueType(hop.getValueType());
      ret.liveOut().addVariable(varName, diVar);
      originalSb.liveIn().addVariable(varName, diVar);
    }

    return ret;
  }

  // process a Hop to TRead and TWrite to be consumed.
  private Hop materialize(Hop hop, ArrayList<DataOp> intermediateWrites) {
    if (!(hop instanceof FunctionOp fop))
      return hop;

    String varName = fop.getOutputVariableNames()[0];

    DataOp tWrite = HopRewriteUtils.createTransientWrite(varName, fop);
    intermediateWrites.add(tWrite);

    return HopRewriteUtils.createTransientRead(varName, fop);
  }

  /**
   * Generate the Hop to replace the existing root.
   * 
   * @param root        root of the current rewrite if `optimalJoin` corresponds
   *                    to the root, otherwise null
   * @param bases
   * @param optimalJoin the current JoinNode we are constructing
   */
  private Hop generateHop(FunctionOp root, ArrayList<DataOp> intermediateWrites, ArrayList<Hop> bases,
      JoinNode optimalJoin) {
    if (optimalJoin instanceof BaseNode baseNode) {
      return bases.get(baseNode.i);
    }
    BinaryNode binaryNode = (BinaryNode) optimalJoin;

    String[] inputNames = new String[] { "A", "colA", "B", "colB", "method" };
    String[] outputNames;
    ArrayList<Hop> outputHops;

    Hop a = generateHop(null, intermediateWrites, bases, binaryNode.left);
    a = materialize(a, intermediateWrites);
    Hop colA = new LiteralOp(binaryNode.leftCol);
    Hop b = generateHop(null, intermediateWrites, bases, binaryNode.right);
    b = materialize(b, intermediateWrites);
    Hop colB = new LiteralOp(binaryNode.rightCol);
    Hop method = new LiteralOp(binaryNode.method);

    ArrayList<Hop> inputs = new ArrayList<>(List.of(a, colA, b, colB, method));
    String varName = "_rajoin_reorder_tmp_" + a.getHopID() + "_" + b.getHopID();
    if (root != null) {
      outputNames = root.getOutputVariableNames();
      outputHops = root.getOutputs();
    } else {
      outputNames = new String[] { varName };
      outputHops = new ArrayList<>();
    }

    FunctionOp fop = new FunctionOp(FunctionOp.FunctionType.DML, ".builtinNS", "m_raJoin", inputNames, inputs,
        outputNames, outputHops);
    fop.setDim2(a.getDim2() + b.getDim2());
    fop.setDataType(DataType.MATRIX);
    fop.setValueType(ValueType.FP64);
    if (root == null) {
      // Return a TRead if it is not the root.
      return materialize(fop, intermediateWrites);
    } 
    return fop;
  }

  /**
   * get a join that is applicable to left and right
   * 
   * @param canonicalJoins
   * @param left           the bitset representing the left side of the raJoin
   * @param right          the bitset representing the right side of the raJoin
   */
  private CanonicalJoin getValidJoin(ArrayList<CanonicalJoin> canonicalJoins, BitSet left, BitSet right) {
    for (CanonicalJoin join : canonicalJoins) {
      if (left.get(join.aBaseIndex) && right.get(join.bBaseIndex)) {
        return join;
      }
    }
    return null;
  }

  private record IntPair(int left, int right) {
  };

  // representation of the dependencies on the bases and its indices for a given
  // raJoin
  private record CanonicalJoin(int aBaseIndex, long acol, int bBaseIndex, long bcol, String method) {
  };

  /**
   * Inorder traversal of an raJoin
   * 
   * @param joinMap
   * @param joins
   * @param canonicaljoins
   * @return inclusive [left, right] range of the indices of `joins` that the
   *         current join corresponds to
   */
  private IntPair dfsInorder(HashMap<String, Integer> joinMap, ArrayList<FunctionOp> joins,
      ArrayList<CanonicalJoin> cannonicalJoins, boolean[] visited, ArrayList<Hop> bases,
      ArrayList<Long> basesLengthPrefixSum, int i) {
    visited[i] = true;
    FunctionOp join = joins.get(i);
    Hop a = join.getInput(0);
    long acol = ((LiteralOp) join.getInput(1)).getLongValue();

    Hop b = join.getInput(2);
    long bcol = ((LiteralOp) join.getInput(3)).getLongValue();

    String method = ((LiteralOp) join.getInput(4)).getStringValue();
    IntPair aPair;
    if (isKnownMatrix(a)) {
      bases.add(a);
      basesLengthPrefixSum
          .add((basesLengthPrefixSum.size() > 0 ? basesLengthPrefixSum.get(basesLengthPrefixSum.size() - 1) : 0)
              + a.getDim2());
      aPair = new IntPair(bases.size() - 1, bases.size() - 1);
    } else {
      Integer aIndex = joinMap.get(a.getName());
      if (aIndex == null)
        throw new UnknownDimensionInfoException();
      aPair = dfsInorder(joinMap, joins, cannonicalJoins, visited, bases, basesLengthPrefixSum, aIndex);
    }
    IntPair bPair;
    if (isKnownMatrix(b)) {
      bases.add(b);
      basesLengthPrefixSum
          .add((basesLengthPrefixSum.size() > 0 ? basesLengthPrefixSum.get(basesLengthPrefixSum.size() - 1) : 0)
              + b.getDim2());
      bPair = new IntPair(bases.size() - 1, bases.size() - 1);
    } else {
      Integer bIndex = joinMap.get(b.getName());
      if (bIndex == null)
        throw new UnknownDimensionInfoException();
      bPair = dfsInorder(joinMap, joins, cannonicalJoins, visited, bases, basesLengthPrefixSum, bIndex);
    }
    int aBaseIndex = -1;
    for (int j = aPair.left; j <= aPair.right; j++) {
      if (acol <= basesLengthPrefixSum.get(j)) {
        // if (j - 1 >= 0) acol -= basesLengthPrefixSum.get(j-1);
        aBaseIndex = j;
        break;
      }
    }

    int bBaseIndex = -1;
    for (int j = bPair.left; j <= bPair.right; j++) {
      if (bcol <= basesLengthPrefixSum.get(j)) {
        // if (j - 1 >= 0) bcol -= basesLengthPrefixSum.get(j-1);
        bBaseIndex = j;
        break;
      }
    }
    acol = getRelativeCol(basesLengthPrefixSum, aPair.left, aBaseIndex, acol);
    bcol = getRelativeCol(basesLengthPrefixSum, bPair.left, bBaseIndex, bcol);

    // throw an error and do not rewrite if we cannot figure out the dependencies.
    if (aBaseIndex < 0 || bBaseIndex < 0) {
      throw new UnknownCanonicalJoinException();
    }
    cannonicalJoins.add(new CanonicalJoin(aBaseIndex, acol, bBaseIndex, bcol, method));
    return new IntPair(aPair.left, bPair.right);
  }

  @Override
  public List<StatementBlock> rewriteStatementBlock(StatementBlock sb, ProgramRewriteStatus state) {
    List<StatementBlock> ret = new ArrayList<>();
    ret.add(sb);
    return ret;
  }

  @Override
  public List<StatementBlock> rewriteStatementBlocks(List<StatementBlock> sbs, ProgramRewriteStatus state) {
    HashMap<Hop, StatementBlock> hopToSb = new HashMap<>();
    HashMap<String, Integer> joinMap = new HashMap<>();
    ArrayList<FunctionOp> joins = new ArrayList<>();
    ArrayList<StatementBlock> sbsA = new ArrayList<>(sbs);
    for (StatementBlock sb : sbsA) {
      collectRaJoin(hopToSb, sb, joinMap, joins);
    }
    try {
      ArrayList<Integer> order = topoOrder(joinMap, joins);
      rewriteRoots(sbsA, hopToSb, joinMap, joins, order);
    } catch (Exception e) {
      // if it is a local exception, try rewriting the next root.
      if (!((e instanceof UnknownCanonicalJoinException) || (e instanceof UnknownDimensionInfoException))) {
        throw e;
      }
    }
    return sbs;
  }
}