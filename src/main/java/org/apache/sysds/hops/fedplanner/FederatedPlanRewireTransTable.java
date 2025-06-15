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

package org.apache.sysds.hops.fedplanner;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.ParamBuiltinOp;
import org.apache.sysds.hops.*;
import org.apache.sysds.hops.FunctionOp.FunctionType;
import org.apache.sysds.parser.*;
import org.apache.sysds.hops.fedplanner.FederatedMemoTable.HopCommon;
import org.apache.sysds.hops.rewrite.HopRewriteUtils;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRange;
import org.apache.sysds.runtime.controlprogram.federated.FederationUtils;
import org.apache.sysds.runtime.util.UtilFunctions;
import java.util.*;
import org.apache.sysds.runtime.instructions.fed.InitFEDInstruction;
import org.apache.sysds.runtime.controlprogram.federated.FederatedData;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.UnknownHostException;
import java.util.concurrent.Future;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse;
import org.apache.sysds.hops.fedplanner.FTypes.Privacy;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.hops.fedplanner.FTypes.FType;
import org.apache.sysds.common.Types.AggOp;
import org.apache.sysds.common.Types.OpOp1;
import org.apache.sysds.common.Types.OpOp2;
import org.apache.sysds.common.Types.OpOp3;
import org.apache.sysds.common.Types.OpOpN;
import org.apache.sysds.common.Types.OpOpDG;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.common.Types.ReOrgOp;
import org.apache.sysds.common.Types.OpOpData;
import org.apache.sysds.lops.MMTSJ.MMTSJType;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;

public class FederatedPlanRewireTransTable {
    
    private static final double DEFAULT_LOOP_WEIGHT = 10.0;
    private static final double DEFAULT_IF_ELSE_WEIGHT = 0.5;

    public static final String FED_MATRIX_IDENTIFIER = "matrix";
    public static final String FED_FRAME_IDENTIFIER = "frame";

    public static void rewireProgram(DMLProgram prog, Map<Long, List<Hop>> rewireTable,
            Map<Long, HopCommon> hopCommonTable, Map<Long, Privacy> privacyConstraintMap, Map<Long, FType> fTypeMap,
            List<Pair<FederatedRange, FederatedData>> fedMap, Set<Long> unRefTwriteSet, Set<Long> unRefSet,
            Set<Hop> progRootHopSet) {
        // Maps Hop ID and fedOutType pairs to their plan variants
        Set<Long> visitedHops = new HashSet<>();
        Set<String> fnStack = new HashSet<>();
        List<Pair<Long, Double>> loopStack = new ArrayList<>();

        List<Map<String, List<Hop>>> outerTransTableList = new ArrayList<>();
        Map<String, List<Hop>> outerTransTable = new HashMap<>();
        outerTransTableList.add(outerTransTable);

        for (StatementBlock sb : prog.getStatementBlocks()) {
            Map<String, List<Hop>> innerTransTable = rewireStatementBlock(sb, prog, visitedHops, rewireTable,
                    hopCommonTable, outerTransTableList, null, privacyConstraintMap, fTypeMap,
                    fedMap, unRefTwriteSet, unRefSet, progRootHopSet, fnStack, 1, 1, loopStack);
            outerTransTableList.get(0).putAll(innerTransTable);
        }
    }

    public static void rewireFunctionDynamic(FunctionStatementBlock function, Map<Long, List<Hop>> rewireTable,
            Map<Long, HopCommon> hopCommonTable, Map<Long, Privacy> privacyConstraintMap, Map<Long, FType> fTypeMap,
            List<Pair<FederatedRange, FederatedData>> fedMap, Set<Long> unRefTwriteSet, Set<Long> unRefSet,
            Set<Hop> progRootHopSet) {
        Set<Long> visitedHops = new HashSet<>();
        Set<String> fnStack = new HashSet<>();
        List<Pair<Long, Double>> loopStack = new ArrayList<>();
        List<Map<String, List<Hop>>> outerTransTableList = new ArrayList<>();
        Map<String, List<Hop>> outerTransTable = new HashMap<>();
        outerTransTableList.add(outerTransTable);
        // Todo (Future): not tested & not used
        rewireStatementBlock(function, null, visitedHops, rewireTable, hopCommonTable, outerTransTableList, null,
                privacyConstraintMap, fTypeMap,
                fedMap, unRefTwriteSet, unRefSet, progRootHopSet, fnStack, 1, 1, loopStack);
    }

    public static Map<String, List<Hop>> rewireStatementBlock(StatementBlock sb, DMLProgram prog, Set<Long> visitedHops,
            Map<Long, List<Hop>> rewireTable, Map<Long, HopCommon> hopCommonTable,
            List<Map<String, List<Hop>>> outerTransTableList, Map<String, List<Hop>> formerTransTable,
            Map<Long, Privacy> privacyConstraintMap, Map<Long, FType> fTypeMap,
            List<Pair<FederatedRange, FederatedData>> fedMap, Set<Long> unRefTwriteSet, Set<Long> unRefSet,
            Set<Hop> progRootHopSet, Set<String> fnStack,
            double computeWeight, double networkWeight, List<Pair<Long, Double>> parentLoopStack) {
        List<Map<String, List<Hop>>> newOuterTransTableList = new ArrayList<>();
        if (outerTransTableList != null) {
            for (Map<String, List<Hop>> outerTable : outerTransTableList) {
                if (outerTable != null && !outerTable.isEmpty()) {
                    newOuterTransTableList.add(outerTable);
                }
            }
        }
        if (formerTransTable != null && !formerTransTable.isEmpty()) {
            newOuterTransTableList.add(formerTransTable);
        }

        Map<String, List<Hop>> newFormerTransTable = new HashMap<>();
        Map<String, List<Hop>> innerTransTable = new HashMap<>();

        if (sb instanceof IfStatementBlock) {
            IfStatementBlock isb = (IfStatementBlock) sb;
            IfStatement istmt = (IfStatement) isb.getStatement(0);

            rewireHopDAG(isb.getPredicateHops(), prog, visitedHops, rewireTable, hopCommonTable, newOuterTransTableList,
                    null, innerTransTable,
                    privacyConstraintMap, fTypeMap, fedMap, unRefTwriteSet, unRefSet, progRootHopSet, fnStack, computeWeight,
                    networkWeight, parentLoopStack);

            newFormerTransTable.putAll(innerTransTable);
            Map<String, List<Hop>> elseFormerTransTable = new HashMap<>();
            elseFormerTransTable.putAll(innerTransTable);
            computeWeight *= DEFAULT_IF_ELSE_WEIGHT;

            for (StatementBlock innerIsb : istmt.getIfBody())
                newFormerTransTable.putAll(rewireStatementBlock(innerIsb, prog, visitedHops, rewireTable,
                        hopCommonTable, newOuterTransTableList, newFormerTransTable,
                        privacyConstraintMap, fTypeMap, fedMap, unRefTwriteSet, unRefSet, progRootHopSet, fnStack, computeWeight,
                        networkWeight, parentLoopStack));

            for (StatementBlock innerIsb : istmt.getElseBody())
                elseFormerTransTable.putAll(rewireStatementBlock(innerIsb, prog, visitedHops, rewireTable,
                        hopCommonTable, newOuterTransTableList, elseFormerTransTable,
                        privacyConstraintMap, fTypeMap, fedMap, unRefTwriteSet, unRefSet, progRootHopSet, fnStack, computeWeight,
                        networkWeight, parentLoopStack));

            // If there are common keys: merge elseValue list into ifValue list
            elseFormerTransTable.forEach((key, elseValue) -> {
                newFormerTransTable.merge(key, elseValue, (ifValue, newValue) -> {
                    ifValue.addAll(newValue);
                    return ifValue;
                });
            });
        } else if (sb instanceof ForStatementBlock) { // incl parfor
            ForStatementBlock fsb = (ForStatementBlock) sb;
            ForStatement fstmt = (ForStatement) fsb.getStatement(0);

            // Calculate for-loop iteration count if possible
            double loopWeight = DEFAULT_LOOP_WEIGHT;
            Hop from = fsb.getFromHops().getInput().get(0);
            Hop to = fsb.getToHops().getInput().get(0);
            Hop incr = (fsb.getIncrementHops() != null) ? fsb.getIncrementHops().getInput().get(0) : new LiteralOp(1);

            // Calculate for-loop iteration count (weight) if from, to, and incr are literal
            // ops (constant values)
            if (from instanceof LiteralOp && to instanceof LiteralOp && incr instanceof LiteralOp) {
                double dfrom = HopRewriteUtils.getDoubleValue((LiteralOp) from);
                double dto = HopRewriteUtils.getDoubleValue((LiteralOp) to);
                double dincr = HopRewriteUtils.getDoubleValue((LiteralOp) incr);
                if (dfrom > dto && dincr == 1)
                    dincr = -1;
                loopWeight = UtilFunctions.getSeqLength(dfrom, dto, dincr, false);
            }
            computeWeight *= loopWeight;
            networkWeight *= loopWeight;

            // Create current loop context (copy parent context)
            List<Pair<Long, Double>> currentLoopStack = new ArrayList<>(parentLoopStack);
            currentLoopStack.add(Pair.of(sb.getSBID(), loopWeight));

            rewireHopDAG(fsb.getFromHops(), prog, visitedHops, rewireTable, hopCommonTable, newOuterTransTableList,
                    null, innerTransTable,
                    privacyConstraintMap, fTypeMap, fedMap, unRefTwriteSet, unRefSet, progRootHopSet, fnStack, computeWeight,
                    networkWeight, currentLoopStack);
            rewireHopDAG(fsb.getToHops(), prog, visitedHops, rewireTable, hopCommonTable, newOuterTransTableList, null,
                    innerTransTable,
                    privacyConstraintMap, fTypeMap, fedMap, unRefTwriteSet, unRefSet, progRootHopSet, fnStack, computeWeight,
                    networkWeight, currentLoopStack);

            if (fsb.getIncrementHops() != null) {
                rewireHopDAG(fsb.getIncrementHops(), prog, visitedHops, rewireTable, hopCommonTable,
                        newOuterTransTableList, null, innerTransTable,
                        privacyConstraintMap, fTypeMap, fedMap, unRefTwriteSet, unRefSet, progRootHopSet, fnStack, computeWeight,
                        networkWeight, currentLoopStack);
            }
            newFormerTransTable.putAll(innerTransTable);

            for (StatementBlock innerFsb : fstmt.getBody())
                newFormerTransTable.putAll(rewireStatementBlock(innerFsb, prog, visitedHops, rewireTable,
                        hopCommonTable, newOuterTransTableList, newFormerTransTable,
                        privacyConstraintMap, fTypeMap, fedMap, unRefTwriteSet, unRefSet, progRootHopSet, fnStack, computeWeight,
                        networkWeight, currentLoopStack));

            // Wire UnRefTwrite to liveOutHops
            wireUnRefTwriteToLiveOut(fsb, unRefTwriteSet, hopCommonTable, newFormerTransTable);
        } else if (sb instanceof WhileStatementBlock) {
            WhileStatementBlock wsb = (WhileStatementBlock) sb;
            WhileStatement wstmt = (WhileStatement) wsb.getStatement(0);

            computeWeight *= DEFAULT_LOOP_WEIGHT;
            networkWeight *= DEFAULT_LOOP_WEIGHT;

            // Create current loop context (copy parent context)
            List<Pair<Long, Double>> currentLoopStack = new ArrayList<>(parentLoopStack);
            currentLoopStack.add(Pair.of(sb.getSBID(), DEFAULT_LOOP_WEIGHT));

            rewireHopDAG(wsb.getPredicateHops(), prog, visitedHops, rewireTable, hopCommonTable, newOuterTransTableList,
                    null, innerTransTable,
                    privacyConstraintMap, fTypeMap, fedMap, unRefTwriteSet, unRefSet, progRootHopSet, fnStack, computeWeight,
                    networkWeight, currentLoopStack);
            newFormerTransTable.putAll(innerTransTable);

            for (StatementBlock innerWsb : wstmt.getBody())
                newFormerTransTable.putAll(rewireStatementBlock(innerWsb, prog, visitedHops, rewireTable,
                        hopCommonTable, newOuterTransTableList, newFormerTransTable,
                        privacyConstraintMap, fTypeMap, fedMap, unRefTwriteSet, unRefSet, progRootHopSet, fnStack, computeWeight,
                        networkWeight, currentLoopStack));

            // Wire UnRefTwrite to liveOutHops
            wireUnRefTwriteToLiveOut(wsb, unRefTwriteSet, hopCommonTable, newFormerTransTable);
        } else if (sb instanceof FunctionStatementBlock) {
            FunctionStatementBlock fsb = (FunctionStatementBlock) sb;
            FunctionStatement fstmt = (FunctionStatement) fsb.getStatement(0);

            for (StatementBlock innerFsb : fstmt.getBody())
                newFormerTransTable.putAll(rewireStatementBlock(innerFsb, prog, visitedHops, rewireTable,
                        hopCommonTable, newOuterTransTableList, newFormerTransTable,
                        privacyConstraintMap, fTypeMap, fedMap, unRefTwriteSet, unRefSet, progRootHopSet, fnStack, computeWeight,
                        networkWeight, parentLoopStack));

            // Wire fcall operation to liveOutHops
            wireUnRefTwriteToLiveOut(fsb, unRefTwriteSet, hopCommonTable, newFormerTransTable);
        } else { // generic (last-level)
            if (sb.getHops() != null) {
                for (Hop c : sb.getHops())
                    rewireHopDAG(c, prog, visitedHops, rewireTable, hopCommonTable, newOuterTransTableList, null,
                            innerTransTable,
                            privacyConstraintMap, fTypeMap, fedMap, unRefTwriteSet, unRefSet, progRootHopSet, fnStack,
                            computeWeight, networkWeight, parentLoopStack);
            }

            return innerTransTable;
        }
        return newFormerTransTable;
    }

    private static void rewireHopDAG(Hop hop, DMLProgram prog, Set<Long> visitedHops, Map<Long, List<Hop>> rewireTable,
            Map<Long, HopCommon> hopCommonTable, List<Map<String, List<Hop>>> outerTransTableList,
            Map<String, List<Hop>> formerTransTable, Map<String, List<Hop>> innerTransTable,
            Map<Long, Privacy> privacyConstraintMap, Map<Long, FType> fTypeMap,
            List<Pair<FederatedRange, FederatedData>> fedMap, Set<Long> unRefTwriteSet, Set<Long> unRefSet,
            Set<Hop> progRootHopSet,
            Set<String> fnStack, double computeWeight, double networkWeight, List<Pair<Long, Double>> loopStack) {

        if (hop.getInput() != null) {
            for (Hop inputHop : hop.getInput()) {
                long inputHopID = inputHop.getHopID();
                if (!visitedHops.contains(inputHopID)) {
                    visitedHops.add(inputHopID);
                    rewireHopDAG(inputHop, prog, visitedHops, rewireTable, hopCommonTable, outerTransTableList,
                            formerTransTable, innerTransTable,
                            privacyConstraintMap, fTypeMap, fedMap, unRefTwriteSet, unRefSet, progRootHopSet, fnStack,
                            computeWeight, networkWeight, loopStack);
                }
            }
        }

        hopCommonTable.put(hop.getHopID(), new HopCommon(hop, computeWeight, networkWeight, 0, loopStack));

        // Identify hops to connect to the root dummy node
        // Connect TWrite pred and u(print) to the root dummy node
        if ((hop instanceof DataOp && (hop.getName().equals("__pred"))) // TWrite "__pred"
                || (hop instanceof UnaryOp && ((UnaryOp) hop).getOp() == Types.OpOp1.PRINT) // u(print)
                || (hop instanceof DataOp && ((DataOp) hop).getOp() == Types.OpOpData.PERSISTENTWRITE)) { // PWrite
            progRootHopSet.add(hop);
        } else if (!(hop instanceof DataOp && ((DataOp) hop).getOp() == Types.OpOpData.TRANSIENTWRITE)
                && hop.getParent().size() == 0) {
            unRefSet.add(hop.getHopID());
        }

        if (hop instanceof FunctionOp) {
            // maintain counters and investigate functions if not seen so far
            FunctionOp fop = (FunctionOp) hop;
            unRefTwriteSet.add(fop.getHopID());

            if (fop.getFunctionType() == FunctionType.DML) {
                String fkey = fop.getFunctionKey();

                if (!fnStack.contains(fkey)) {
                    fnStack.add(fkey);
                    FunctionStatementBlock fsb = prog.getFunctionStatementBlock(fop.getFunctionNamespace(),
                            fop.getFunctionName());

                    Map<String, List<Hop>> newFormerTransTable = new HashMap<>();
                    if (formerTransTable != null) {
                        newFormerTransTable.putAll(formerTransTable);
                    }
                    newFormerTransTable.putAll(innerTransTable);

                    String[] inputArgs = fop.getInputVariableNames();
                    List<Hop> inputHops = fop.getInput();

                    // Only used outside of functionTransTable.
                    for (int i = 0; i < inputHops.size(); i++) {
                        newFormerTransTable.computeIfAbsent(inputArgs[i], k -> new ArrayList<>()).add(inputHops.get(i));
                    }

                    Map<String, List<Hop>> functionTransTable = rewireStatementBlock(fsb, prog, visitedHops,
                            rewireTable, hopCommonTable, outerTransTableList, newFormerTransTable,
                            privacyConstraintMap, fTypeMap, fedMap, unRefTwriteSet, unRefSet, progRootHopSet, fnStack,
                            computeWeight, networkWeight, loopStack);

                    for (int i = 0; i < fop.getOutputVariableNames().length; i++) {
                        String tWriteName = fop.getOutputVariableNames()[i];
                        List<Hop> outputHops = functionTransTable.get(fsb.getOutputsofSB().get(i).getName());
                        innerTransTable.computeIfAbsent(tWriteName, k -> new ArrayList<>()).addAll(outputHops);
                        for (Hop outputHop : outputHops) {
                            unRefTwriteSet.add(outputHop.getHopID());
                        }
                    }
                }
            }
        }

        // Propagate Privacy Constraint
        if (!(hop instanceof DataOp) || hop.getName().equals("__pred")
                || (((DataOp) hop).getOp() == Types.OpOpData.PERSISTENTWRITE)) {
            privacyConstraintMap.put(hop.getHopID(),
                    getPrivacyConstraint(hop, hop.getInput(), privacyConstraintMap));
            fTypeMap.put(hop.getHopID(), getFederatedType(hop, fTypeMap));
            // Todo: Remove this after debugging
//            FederatedPlannerLogger.logHopInfo(hop, privacyConstraintMap, fTypeMap, "RewireTransHop");
            return;
        }

        rewireTransHop(hop, rewireTable, outerTransTableList, formerTransTable, innerTransTable, privacyConstraintMap,
                fTypeMap, fedMap, unRefTwriteSet);
        // Todo: Remove this after debugging
//        FederatedPlannerLogger.logHopInfo(hop, privacyConstraintMap, fTypeMap, "RewireTransHop");
    }

    private static void rewireTransHop(Hop hop, Map<Long, List<Hop>> rewireTable,
            List<Map<String, List<Hop>>> outerTransTableList, Map<String, List<Hop>> formerTransTable,
            Map<String, List<Hop>> innerTransTable, Map<Long, Privacy> privacyConstraintMap,
            Map<Long, FType> fTypeMap, List<Pair<FederatedRange, FederatedData>> fedMap, Set<Long> unRefTwriteSet) {
        DataOp dataOp = (DataOp) hop;
        Types.OpOpData opType = dataOp.getOp();
        String hopName = dataOp.getName();

        if (opType == Types.OpOpData.FEDERATED) {
            Privacy privacy = getFedWorkerMetaData(fedMap, dataOp);
            privacyConstraintMap.put(hop.getHopID(), privacy);
            FType fType = deriveFType((DataOp)hop);
            fTypeMap.put(hop.getHopID(), fType);
        } else if (opType == Types.OpOpData.TRANSIENTWRITE) {
            // Rewire TransWrite
            innerTransTable.computeIfAbsent(hopName, k -> new ArrayList<>()).add(hop);
            unRefTwriteSet.add(hop.getHopID());
            // Propagate Privacy Constraint
            privacyConstraintMap.put(hop.getHopID(),
                    getPrivacyConstraint(hop, hop.getInput(), privacyConstraintMap));         
            // Propagate FType (TransWrite has only one input)
            FType inputFType = fTypeMap.get(hop.getInput(0).getHopID());
            fTypeMap.put(hop.getHopID(), inputFType);
        } else if (opType == Types.OpOpData.TRANSIENTREAD) {
            // Rewire TransRead
            List<Hop> childHops = rewireTransRead(hopName, innerTransTable, formerTransTable, outerTransTableList);
            // Handle rewire table (TransRead -> TransWrite)
            rewireTable.put(hop.getHopID(), childHops);

            // Todo: Handle exception when TRead has no Child (check why it's missing)
            if (childHops == null || childHops.isEmpty()) {
                FederatedPlannerLogger.logTransReadRewireDebug(hopName, hop.getHopID(), childHops, true, "RewireTransHop");
                return;
            }

            // Remove childHops that have different hopVarName
            List<Hop> filteredChildHops = new ArrayList<>();
            for (Hop childHop : childHops) {
                String hopVarName = hop.getName();

                if (hopVarName.equals(childHop.getName())) {
                    filteredChildHops.add(childHop);
                }
            }

            // Todo: Handle exception when TRead has no Filtered Child (check why it's missing)
            if (filteredChildHops.isEmpty()) {
                FederatedPlannerLogger.logFilteredChildHopsDebug(hopName, hop.getHopID(), filteredChildHops, true, "RewireTransHop");
                return;
            }

            FType inputFType = null;
            for (int i = 0; i < filteredChildHops.size(); i++) {
                Hop filteredChildHop = filteredChildHops.get(i);
                long filteredChildHopID = filteredChildHop.getHopID();

                // Rewire (TransWrite -> TransRead)
                rewireTable.computeIfAbsent(filteredChildHopID, k -> new ArrayList<>()).add(hop);
                // Remove refTWrite from unRefTwriteSet
                unRefTwriteSet.remove(filteredChildHopID);

                // Check FType consistency of childs(TransWrite)
                if ( i==0 ) {
                    inputFType = fTypeMap.get(filteredChildHopID);
                } else if (inputFType != fTypeMap.get(filteredChildHopID)) {
                    // Todo: Handle exception when TRead has different FType
                    FType mismatchedFType = fTypeMap.get(filteredChildHopID);
                    FederatedPlannerLogger.logFTypeMismatchError(hop, filteredChildHops, fTypeMap, inputFType, mismatchedFType, i);
                    
                    if (inputFType == null) {
                        inputFType = mismatchedFType;
                    }
                    // throw new DMLRuntimeException("TransRead input FType mismatch: " + inputFType + " != " + mismatchedFType);
                }
            }
            // Propagate Privacy Constraint
            privacyConstraintMap.put(hop.getHopID(),
                    getPrivacyConstraint(hop, filteredChildHops, privacyConstraintMap));
            // Propagate FType
            fTypeMap.put(hop.getHopID(), inputFType);
        } else {
            privacyConstraintMap.put(hop.getHopID(),
                    getPrivacyConstraint(hop, hop.getInput(), privacyConstraintMap));
            fTypeMap.put(hop.getHopID(), getFederatedType(hop, fTypeMap));
        }
    }

    private static List<Hop> rewireTransRead(String hopName, Map<String, List<Hop>> innerTransTable,
            Map<String, List<Hop>> formerTransTable, List<Map<String, List<Hop>>> outerTransTableList) {
        List<Hop> childHops = new ArrayList<>();

        // Read according to priority: inner -> former -> outer
        if (!innerTransTable.isEmpty()) {
            childHops = innerTransTable.get(hopName);
        }

        if ((childHops == null || childHops.isEmpty()) && formerTransTable != null) {
            childHops = formerTransTable.get(hopName);
        }

        if (childHops == null || childHops.isEmpty()) {
            // Traverse in reverse order from the last inserted outerTransTable
            for (int i = outerTransTableList.size() - 1; i >= 0; i--) {
                Map<String, List<Hop>> outerTransTable = outerTransTableList.get(i);
                childHops = outerTransTable.get(hopName);
                if (childHops != null && !childHops.isEmpty())
                    break;
            }
        }

        return childHops;
    }

    private static Privacy getFedWorkerMetaData(List<Pair<FederatedRange, FederatedData>> fedMap, DataOp initFedOp) {
        // Address
        Hop addressListHop = initFedOp.getInput(initFedOp.getParameterIndex("addresses"));
        List<String> addressList = new ArrayList<>();
        for (Hop addressHop : addressListHop.getInput()) {
            addressList.add(addressHop.getName());
        }

        // Range
        Hop rangeListHop = initFedOp.getInput(initFedOp.getParameterIndex("ranges"));
        List<long[]> rangeList = new ArrayList<>();
        for (Hop rangeHop : rangeListHop.getInput()) {
            long beginRange = (long) Double.parseDouble(rangeHop.getInput(0).getName());
            long endRange = (long) Double.parseDouble(rangeHop.getInput(1).getName());
            rangeList.add(new long[] { beginRange, endRange });
        }

        // Type
        String type = initFedOp.getInput(initFedOp.getParameterIndex("type")).getName();
        Types.DataType fedDataType;

        if (type.equalsIgnoreCase(FED_MATRIX_IDENTIFIER))
            fedDataType = Types.DataType.MATRIX;
        else
            fedDataType = Types.DataType.FRAME;

        // Init Fed Data
        for (int i = 0; i < addressList.size(); i++) {
            String address = addressList.get(i);
            // We split address into url/ip, the port and file path of file to read
            String[] parsedValues = InitFEDInstruction.parseURL(address);
            String host = parsedValues[0];
            int port = Integer.parseInt(parsedValues[1]);
            String filePath = parsedValues[2];

            long[] beginRange = rangeList.get(2 * i);
            long[] endRange = rangeList.get(2 * i + 1);

            try {
                FederatedData federatedData = new FederatedData(fedDataType,
                        new InetSocketAddress(InetAddress.getByName(host), port), filePath);
                fedMap.add(new ImmutablePair<>(new FederatedRange(beginRange, endRange), federatedData));
            } catch (UnknownHostException e) {
                throw new RuntimeException("federated host was unknown: " + host, e);
            }
        }
        Privacy privacyConstraint = null;

        // Request Privacy Constraints
        for (Pair<FederatedRange, FederatedData> fed : fedMap) {
            FederatedData data = fed.getRight();
            data.initFederatedData(FederationUtils.getNextFedDataID());

            Future<FederatedResponse> future = data.requestPrivacyConstraints();
            try {
                FederatedResponse response = future.get(); // Get actual response from Future

                if (response.isSuccessful()) {
                    Object[] responseData = response.getData();
                    String privacyConstraints = (String) responseData[0]; // Cast privacy constraint as string
                    String pcLower = privacyConstraints.trim().toLowerCase();
                    Privacy tempPrivacy = null;

                    // Map to appropriate PrivacyConstraint value based on input string
                    if (pcLower.equals("private")
                            || pcLower.equals(FTypes.Privacy.PRIVATE.toString().toLowerCase())) {
                        tempPrivacy = FTypes.Privacy.PRIVATE;
                    } else if (pcLower.equals("private-aggregate") || pcLower.equals("private_aggregate") ||
                            pcLower.equals(FTypes.Privacy.PRIVATE_AGGREGATE.toString().toLowerCase())) {
                        tempPrivacy = FTypes.Privacy.PRIVATE_AGGREGATE;
                    } else if (pcLower.equals("public")
                            || pcLower.equals(FTypes.Privacy.PUBLIC.toString().toLowerCase())) {
                        tempPrivacy = FTypes.Privacy.PUBLIC;
                    } else {
                        throw new DMLRuntimeException("Invalid privacy constraint: " + privacyConstraints +
                                ". Must be one of 'PRIVATE', 'PRIVATE_AGGREGATE', 'PUBLIC'.");
                    }

                    if (privacyConstraint == null) {
                        privacyConstraint = tempPrivacy;
                    } else {
                        if (privacyConstraint != tempPrivacy) {
                            throw new DMLRuntimeException("Privacy constraints do not match.");
                        }
                    }
                } else {
                    // Error handling
                    String errorMsg = response.getErrorMessage();
                    System.err.println("Failed to request privacy constraints: " + errorMsg);
                }
            } catch (Exception e) {
                // Exception handling
                e.printStackTrace();
            }
        }
        return privacyConstraint;
    }

    private static Privacy getPrivacyConstraint(Hop hop, List<Hop> inputHops, Map<Long, Privacy> privacyMap) {
        Privacy[] pc = new Privacy[inputHops.size()];
        for (int i = 0; i < inputHops.size(); i++)
            pc[i] = privacyMap.get(inputHops.get(i).getHopID());

        boolean hasPrivateAggreate = false;

        for (Privacy p : pc) {
            if (p == Privacy.PRIVATE) {
                return Privacy.PRIVATE;
            } else if (p == Privacy.PRIVATE_AGGREGATE) {
                hasPrivateAggreate = true;
            }
        }

        if (hasPrivateAggreate) {
            if (hop instanceof AggUnaryOp || hop instanceof AggBinaryOp || hop instanceof QuaternaryOp) {
                return Privacy.PUBLIC;
            } else if (hop instanceof TernaryOp) {
                switch (((TernaryOp) hop).getOp()) {
                    case MOMENT:
                    case COV:
                    case CTABLE:
                    case INTERQUANTILE:
                    case QUANTILE:
                        return Privacy.PUBLIC;
                    default:
                        return Privacy.PRIVATE_AGGREGATE;
                }
            } else if (hop instanceof ParameterizedBuiltinOp
                    && ((ParameterizedBuiltinOp) hop).getOp() == ParamBuiltinOp.GROUPEDAGG) {
                return Privacy.PUBLIC;
            } else {
                return Privacy.PRIVATE_AGGREGATE;
            }
        }

        return Privacy.PUBLIC;
    }

    /**
     * Determines the federated partition type (FType) for the output of a given hop operation.
     * This method combines the logic of checking federated support and determining output FType.
     * 
     * @param hop The hop operation to analyze
     * @param fTypeMap Map containing FType information for all processed hops
     * @return The FType of the output, or null if the operation doesn't support federated execution
     *         or produces non-federated output
     */
    private static FType getFederatedType(Hop hop, Map<Long, FType> fTypeMap) {
        // ========================================================================
        // PART 1: Universal constraints - operations that NEVER support federated
        // ========================================================================
        
        // Scalar values don't have FType (no partitioning concept for scalars)
        if (hop.isScalar()) {
            return null;
        }
        
        // Operations architecturally incompatible with federated execution:
        // - DataGenOp: All data generation requires centralized execution (RAND seed sync, SEQ global coords, etc.)
        // - DnnOp: Deep learning operations designed exclusively for CP/GPU (CuDNN dependencies)
        // - FunctionOp: Function calls execute locally on coordinator (no 'fcall' in FEDInstructionParser)
        // - LiteralOp: Constants without computation, created at coordinator only
        // - DataOp: Data operations (FEDERATED, TRANSIENTREAD, TRANSIENTWRITE) 은 따로 처리, 나머지는 지원 안함 (PERSISTENTWRITE/READ, FUNCTIONOUTPUT, SQLREAD)
        if (hop instanceof DataGenOp || hop instanceof DnnOp || 
            hop instanceof FunctionOp || hop instanceof LiteralOp ||
            hop instanceof DataOp) {
            return null;
        }

        // Extract input FTypes for analysis
        FType[] ft = new FType[hop.getInput().size()];
        for (int i = 0; i < hop.getInput().size(); i++)
            ft[i] = fTypeMap.get(hop.getInput(i).getHopID());

        // Handle operations with no inputs
        if (ft.length == 0) {
            return null;
        }

        // Common patterns used across multiple operation types
        FType firstFType = ft[0];
        boolean hasFederatedFirstInput = firstFType != null;

        // ========================================================================
        // PART 2: Operations NOT requiring federated first input
        // ========================================================================
        
        // NaryOp: N-ary operations with matrix/list support
        if (hop instanceof NaryOp) {
            OpOpN op = ((NaryOp) hop).getOp();
            
            // Unsupported operations:
            // - PRINTF/EVAL: Output operations, execute on coordinator only
            // - LIST: List operations not federated
            // - CBIND/RBIND on lists: Only matrix operations supported
            // - Cell operations on all scalars: No distribution benefit
            if (op == OpOpN.PRINTF || op == OpOpN.EVAL || op == OpOpN.LIST ||
                ((op == OpOpN.CBIND || op == OpOpN.RBIND) && 
                hop.getInput().get(0).getDataType().isList()) ||
                (op.isCellOp() && 
                hop.getInput().stream().allMatch(h -> h.getDataType().isScalar()))) {
                return null;
            }
            
            // Supported matrix operations: CBIND/RBIND (matrix concat), PLUS/MULT (arithmetic), MIN/MAX (comparison)
            if (op == OpOpN.CBIND || op == OpOpN.RBIND ||
                op == OpOpN.PLUS || op == OpOpN.MULT ||
                op == OpOpN.MIN || op == OpOpN.MAX) {
                FType secondFType = ft.length > 1 ? ft[1] : null;
                return firstFType != null ? firstFType : secondFType;
            }
            
            // Other NaryOp operations not supported
            return null;
        }
        
        // TernaryOp: Three-input operations with complex federation patterns
        if (hop instanceof TernaryOp) {
            // Scalar output operations don't have FType
            if (hop.getDataType().isScalar()) {
                return null;
            }
            
            // Operations that produce scalar output or are unsupported:
            // - MOMENT/COV: Aggregation operations produce scalar
            // - IFELSE/MAP: No federated implementation
            OpOp3 op = ((TernaryOp) hop).getOp();
            if (op == OpOp3.MOMENT || op == OpOp3.COV ||
                op == OpOp3.IFELSE || op == OpOp3.MAP) {
                return null;
            }

            // Check if any input is federated
            boolean hasAnyFederatedInput = false;
            boolean hasRowPartition = false;
            for (FType f : ft) {
                if (f == FType.ROW) {
                    hasRowPartition = true;
                    hasAnyFederatedInput = true;
                    break;
                } else if (f != null) {
                    hasAnyFederatedInput = true;
                }
            }
            
            // Requires at least one federated input
            // CTABLE: Special ROW partition requirement
            if (!hasAnyFederatedInput || (!hasRowPartition && op == OpOp3.CTABLE)) {
                return null;
            }

            // All supported operations propagate first non-null FType
            FType secondFType = ft.length > 1 ? ft[1] : null;
            return firstFType != null ? firstFType : 
                secondFType != null ? secondFType : 
                ft.length > 2 ? ft[2] : null;
        }
        
        // AggBinaryOp: Matrix multiplication and aggregation operations
        if (hop instanceof AggBinaryOp) {
            FType secondFType = ft.length > 1 ? ft[1] : null;
            boolean hasFederatedSecondInput = secondFType != null;
            
            // Check supported federation patterns
            if(!((hasFederatedFirstInput != hasFederatedSecondInput) || // One federated, one not
            (firstFType != null && firstFType == secondFType) || // Both federated with same type
            (firstFType == FType.COL && secondFType == FType.ROW))) { // Special matrix multiplication patterns
                return null;
            }

            // Determine output FType based on operation type
            MMTSJType mmtsj = ((AggBinaryOp) hop).checkTransposeSelf();
            
            // Self-transpose matrix multiplication (X'X or XX') results in BROADCAST
            if (mmtsj != MMTSJType.NONE &&
                ((mmtsj.isLeft() && firstFType == FType.ROW) || 
                (mmtsj.isRight() && firstFType == FType.COL))) {
                return FType.BROADCAST;
            }
            // One federated input: propagate its FType
            else if ((firstFType != null) != (secondFType != null)) {
                return firstFType != null ? firstFType : secondFType;
            }
            // COL x ROW multiplication results in ROW partitioning
            else if (firstFType == FType.COL && secondFType == FType.ROW) {
                return FType.ROW;
            }
            // Same partition type: maintain it
            else if ((firstFType == FType.ROW && secondFType == FType.ROW) || 
                    (firstFType == FType.COL && secondFType == FType.COL)) {
                return firstFType;
            }
            return null;
        }
        
        // BinaryOp: Standard binary operations (+, -, *, /, min, max)
        if (hop instanceof BinaryOp) {
            // Scalar operations don't have FType
            if (hop.getDataType().isScalar()) {
                return null;
            }
            
            FType secondFType = ft.length > 1 ? ft[1] : null;
            boolean hasFederatedSecondInput = secondFType != null;
            
            // Unsupported patterns: no federated inputs, or both federated with different types
            if ((!hasFederatedFirstInput && !hasFederatedSecondInput) ||
                (hasFederatedFirstInput && hasFederatedSecondInput && firstFType != secondFType)) {
                return null;
            }

            // Propagate first non-null FType
            return firstFType != null ? firstFType : secondFType;
        }

        // ========================================================================
        // PART 3: Operations REQUIRING federated first input
        // ========================================================================
        
        // All remaining operations require federated first input
        if (!hasFederatedFirstInput) {
            return null;
        }
        
        // Simple operations that maintain input structure:
        // - IndexingOp: Right indexing X[i:j, k:l] - subset of federated matrix remains federated
        // - LeftIndexingOp: Left-hand side indexing X[i:j, k:l] = Y - updates preserve partitioning
        if (hop instanceof IndexingOp || hop instanceof LeftIndexingOp) {
            return firstFType;
        }
        
        // UnaryOp: Element-wise unary operations
        if (hop instanceof UnaryOp) {
            UnaryOp uop = (UnaryOp) hop;
            OpOp1 op = uop.getOp();
            
            // Unsupported operations:
            // - Output operations (PRINT, ASSERT, STOP): Execute on coordinator
            // - Type/metadata operations (TYPEOF, NROW, NCOL): Return scalars
            // - Complex decompositions (INVERSE, EIGEN, etc.): CP-only algorithms
            // - SQRT_MATRIX_JAVA: Special matrix square root, CP only
            // - List operations: List datatype not federated
            // - Metadata operations: Return scalar metadata
            if (op == OpOp1.PRINT || op == OpOp1.ASSERT || op == OpOp1.STOP ||
                op == OpOp1.TYPEOF || op == OpOp1.INVERSE || op == OpOp1.EIGEN ||
                op == OpOp1.CHOLESKY || op == OpOp1.DET || op == OpOp1.SVD ||
                op == OpOp1.SQRT_MATRIX_JAVA ||
                hop.getInput().get(0).getDataType() == DataType.LIST ||
                uop.isMetadataOperation()) {
                return null;
            }
            
            // Element-wise operations maintain structure
            return firstFType;
        }
        
        // QuaternaryOp: Four-input weighted operations
        if (hop instanceof QuaternaryOp) {
            Types.OpOp4 op = ((QuaternaryOp) hop).getOp();
            
            // Scalar output operations:
            // - WSLOSS: Weighted squared loss (returns scalar loss value)
            // - WCEMM: Weighted cross entropy (returns scalar loss value)
            if (op == Types.OpOp4.WSLOSS || op == Types.OpOp4.WCEMM) {
                return null;
            }
            
            // Operations maintaining first input's structure:
            // - WSIGMOID: Weighted sigmoid
            // - WUMM: Weighted unary matrix multiplication
            if (op == Types.OpOp4.WSIGMOID || op == Types.OpOp4.WUMM) {
                return firstFType;
            }
            
            // WDIVMM: Weighted division matrix multiplication - use first non-null FType
            if (op == Types.OpOp4.WDIVMM) {
                FType firstNonNullFType = null;
                for (FType f : ft) {
                    if (f != null) {
                        firstNonNullFType = f;
                        break;
                    }
                }
                return firstNonNullFType;
            }
            
            // Default: maintain first input's FType
            return firstFType;
        }
        
        // AggUnaryOp: Aggregate unary operations with direction awareness
        if (hop instanceof AggUnaryOp) {
            AggOp aggOp = ((AggUnaryOp)hop).getOp();
            
            // Check if aggregation OpCode is supported
            // Supported: SUM, MIN, MAX, SUM_SQ, MEAN, VAR, MAXINDEX, MININDEX
            if (!(aggOp == AggOp.SUM || aggOp == AggOp.MIN || aggOp == AggOp.MAX 
                || aggOp == AggOp.SUM_SQ || aggOp == AggOp.MEAN || aggOp == AggOp.VAR
                || aggOp == AggOp.MAXINDEX || aggOp == AggOp.MININDEX)) {
                return null;
            }
            
            // Determine output FType based on aggregation direction
            boolean isColAgg = ((AggUnaryOp) hop).getDirection().isCol();
            
            // Full aggregation produces scalar result:
            // - ROW partition + column aggregation → scalar per row → local result
            // - COL partition + row aggregation → scalar per column → local result
            if ((firstFType == FType.ROW && isColAgg) || 
                (firstFType == FType.COL && !isColAgg)) {
                return null;
            }
            
            // Partial aggregation maintains structure:
            // - ROW partition + row aggregation → maintains ROW
            // - COL partition + column aggregation → maintains COL
            if (firstFType == FType.ROW || firstFType == FType.COL) {
                return firstFType;
            }
            
            // Other FTypes (FULL, BROADCAST) not affected by direction
            return null;
        }
        
        // ReorgOp: Reorganization operations that transform structure
        if (hop instanceof ReorgOp) {
            ReOrgOp op = ((ReorgOp)hop).getOp();
            
            // Unsupported operations:
            // - RESHAPE: Dimension changes break partitioning assumptions
            // - SORT: Requires global ordering across all partitions
            if (op == ReOrgOp.RESHAPE || op == ReOrgOp.SORT) {
                return null;
            }
            
            // TRANS: Transpose swaps ROW↔COL partitioning
            if (op == ReOrgOp.TRANS) {
                if (firstFType == FType.ROW) return FType.COL;
                if (firstFType == FType.COL) return FType.ROW;
                return firstFType; // FULL/BROADCAST unchanged
            }
            
            // Structure-maintaining operations: DIAG, REV, ROLL
            return firstFType;
        }
        
        // ParameterizedBuiltinOp: Builtin operations with parameters
        if (hop instanceof ParameterizedBuiltinOp) {
            ParamBuiltinOp op = ((ParameterizedBuiltinOp) hop).getOp();
            
            // CONTAINS returns scalar boolean result
            if (op == ParamBuiltinOp.CONTAINS) {
                return null;
            }
            
            // Check if operation is supported
            // Supported: REPLACE, RMEMPTY, LOWER_TRI, UPPER_TRI, TRANSFORMDECODE, TRANSFORMAPPLY, TOKENIZE
            if (!(op == ParamBuiltinOp.REPLACE || op == ParamBuiltinOp.RMEMPTY ||
                op == ParamBuiltinOp.LOWER_TRI || op == ParamBuiltinOp.UPPER_TRI ||
                op == ParamBuiltinOp.TRANSFORMDECODE || op == ParamBuiltinOp.TRANSFORMAPPLY ||
                op == ParamBuiltinOp.TOKENIZE)) {
                return null;
            }
            
            // Structure-preserving operations maintain input FType
            return firstFType;
        }

        // Default: Unknown operation type or unhandled case
        return null;
    }

	private static FType deriveFType(DataOp fedInit) {
		Hop ranges = fedInit.getInput(fedInit.getParameterIndex(DataExpression.FED_RANGES));
		boolean rowPartitioned = true;
		boolean colPartitioned = true;
		for( int i=0; i<ranges.getInput().size()/2; i++ ) { // workers
			Hop beg = ranges.getInput(2*i);
			Hop end = ranges.getInput(2*i+1);
			long rl = HopRewriteUtils.getIntValueSafe(beg.getInput(0));
			long ru = HopRewriteUtils.getIntValueSafe(end.getInput(0));
			long cl = HopRewriteUtils.getIntValueSafe(beg.getInput(1));
			long cu = HopRewriteUtils.getIntValueSafe(end.getInput(1));
			rowPartitioned &= (cu-cl == fedInit.getDim2());
			colPartitioned &= (ru-rl == fedInit.getDim1());
		}
		return rowPartitioned && colPartitioned ?
			FType.FULL : rowPartitioned ? FType.ROW :
			colPartitioned ? FType.COL : FType.OTHER;
	}

    private static void wireUnRefTwriteToLiveOut(StatementBlock sb, Set<Long> unRefTwriteSet,
            Map<Long, HopCommon> hopCommonTable, Map<String, List<Hop>> newFormerTransTable) {
        if (unRefTwriteSet.isEmpty())
            return;

        VariableSet genHops = sb.getGen();
        VariableSet updatedHops = sb.variablesUpdated();
        VariableSet liveOutHops = sb.liveOut();

        Iterator<Long> unRefTwriteIterator = unRefTwriteSet.iterator();
        while (unRefTwriteIterator.hasNext()) {
            Long unRefTwriteHopID = unRefTwriteIterator.next();
            Hop unRefTwriteHop = hopCommonTable.get(unRefTwriteHopID).getHopRef();
            String unRefTwriteHopName = unRefTwriteHop.getName();

            if (liveOutHops.containsVariable(unRefTwriteHopName)) {
                continue;
            }

            if (unRefTwriteHop instanceof FunctionOp || genHops.containsVariable(unRefTwriteHopName) || updatedHops.containsVariable(unRefTwriteHopName)) {
                Iterator<String> liveOutHopsIterator = liveOutHops.getVariableNames().iterator();

                boolean isRewired = false;
                while (liveOutHopsIterator.hasNext()) {
                    String liveOutHopName = liveOutHopsIterator.next();
                    List<Hop> liveOutHopsList = newFormerTransTable.get(liveOutHopName);

                    if (liveOutHopsList != null && !liveOutHopsList.isEmpty()) {
                        List<Hop> copyLiveOutHopsList = new ArrayList<>(liveOutHopsList);
                        copyLiveOutHopsList.add(unRefTwriteHop);
                        newFormerTransTable.put(liveOutHopName, copyLiveOutHopsList);
                        unRefTwriteIterator.remove();
                        isRewired = true;
                        break;
                    }
                }
                if (!isRewired) {
                    throw new RuntimeException("No liveOutHops found for " + unRefTwriteHopName);
                }
            }
        }
    }
}
