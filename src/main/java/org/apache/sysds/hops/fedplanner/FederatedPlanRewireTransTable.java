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
    
    // Enum for standardized reason codes
    public enum ReasonCode {
        // Hop Patterns
        TSMM_PATTERN,
        SCALAR_HOP,
        DISALLOWED_OP,
        
        // AggUnaryOp
        AGGR_UNARY_ALLOWED,
        AGGR_UNARY_DISALLOWED,
        AGGR_DIRECTION_MISMATCH,
        
        // AggBinaryOp
        AGGR_BINARY_MIXED_NULL,
        AGGR_BINARY_COL_ROW,
        AGGR_BINARY_PROPAGATE,
        
        // UnaryOp
        UNARY_DISALLOWED_OP,
        UNARY_LIST_INPUT,
        UNARY_METADATA_OP,
        UNARY_ALLOWED,
        
        // BinaryOp
        BINARY_MIN_DISALLOWED,
        BINARY_MIXED_NULL,
        BINARY_SAME_FTYPE,
        
        // TernaryOp
        TERNARY_CTABLE_IFELSE,
        TERNARY_AT_LEAST_ONE_NON_NULL,
        
        // ReorgOp
        REORG_TRANS_COL_ROW,
        REORG_TRANS_INVALID,
        
        // DataOp
        DATA_FEDERATED,
        DATA_TRANSIENT_WRITE,
        DATA_TRANSIENT_READ,
        
        // FunctionOp
        FUNCTION_TRANSFORM_ENCODE,
        FUNCTION_ALLOWED,
        
        // NaryOp
        NARY_DISALLOWED_OP,
        NARY_LIST_BIND,
        NARY_ALLOWED,
        
        // ParameterizedBuiltinOp
        PARAM_BUILTIN_DISALLOWED,
        PARAM_BUILTIN_ALLOWED,
        
        // DataGenOp
        DATAGEN_DISALLOWED,
        DATAGEN_ALLOWED,
        
        // DnnOp
        DNN_ALWAYS_DISALLOWED,
        
        // Other
        PROPAGATE_FROM_INPUT,
        DERIVED_FROM_FED_RANGES,
        FIRST_NON_NULL,
        UNKNOWN_HOP_TYPE
    }
    
    // Enhanced logging data structure
    public static class EnhancedLogData {
        public final long hopID;
        public final String hopName;
        public final String hopType;
        public final String opCode;
        public final LocalDateTime timestamp;
        public final int callStackDepth;
        public final String stage;
        public final boolean allowsFederated;
        public final FType resultFType;
        public final ReasonCode reasonCode;
        public final String[] inputFTypes;
        public final long[] inputHopIDs;
        public final String[] inputNames;
        public final long[] dimensions;
        public final boolean isSparse;
        public final String[] conditions;
        public final String selectedBranch;
        public final int alternativePaths;
        
        public EnhancedLogData(long hopID, String hopName, String hopType, String opCode,
                LocalDateTime timestamp, int callStackDepth, String stage,
                boolean allowsFederated, FType resultFType, ReasonCode reasonCode,
                String[] inputFTypes, long[] inputHopIDs, String[] inputNames,
                long[] dimensions, boolean isSparse, String[] conditions,
                String selectedBranch, int alternativePaths) {
            this.hopID = hopID;
            this.hopName = hopName;
            this.hopType = hopType;
            this.opCode = opCode;
            this.timestamp = timestamp;
            this.callStackDepth = callStackDepth;
            this.stage = stage;
            this.allowsFederated = allowsFederated;
            this.resultFType = resultFType;
            this.reasonCode = reasonCode;
            this.inputFTypes = inputFTypes;
            this.inputHopIDs = inputHopIDs;
            this.inputNames = inputNames;
            this.dimensions = dimensions;
            this.isSparse = isSparse;
            this.conditions = conditions;
            this.selectedBranch = selectedBranch;
            this.alternativePaths = alternativePaths;
        }
    }
    
    private static final DateTimeFormatter TIMESTAMP_FORMAT = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss.SSS");
    private static final ThreadLocal<Integer> callStackDepth = ThreadLocal.withInitial(() -> 0);
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

            // 현재 루프 컨텍스트 생성 (부모 컨텍스트 복사)
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

            // 현재 루프 컨텍스트 생성 (부모 컨텍스트 복사)
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
        
        logStepState("rewireHopDAG_START", hop, fTypeMap);
        // Process all input nodes first if not already in memo table

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

                    // functionTransTable에서 밖에 안 씀.
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
            
            logStepState("allowsFederated_CHECK", hop, fTypeMap);
            
            if (allowsFederated(hop, fTypeMap)) {
                logStepState("getFType_START", hop, fTypeMap);
                FType resultFType = getFType(hop, fTypeMap);
                fTypeMap.put(hop.getHopID(), resultFType);
                
                logEnhancedFType(hop, true, resultFType, ReasonCode.PROPAGATE_FROM_INPUT, fTypeMap);
            } else {
                fTypeMap.put(hop.getHopID(), null);
                
                logEnhancedFType(hop, false, null, ReasonCode.DISALLOWED_OP, fTypeMap);
            }
            
            logStepState("rewireHopDAG_END", hop, fTypeMap);
            return;
        }

        rewireTransHop(hop, rewireTable, outerTransTableList, formerTransTable, innerTransTable, privacyConstraintMap,
                fTypeMap, fedMap, unRefTwriteSet);
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
            
            logEnhancedFType(hop, true, fType, ReasonCode.DERIVED_FROM_FED_RANGES, fTypeMap);
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
            
            logEnhancedFType(hop, true, inputFType, ReasonCode.PROPAGATE_FROM_INPUT, fTypeMap);
        } else if (opType == Types.OpOpData.TRANSIENTREAD) {
            // Rewire TransRead
            List<Hop> childHops = rewireTransRead(hopName, innerTransTable, formerTransTable, outerTransTableList);
            // Handle rewire table (TransRead -> TransWrite)
            rewireTable.put(hop.getHopID(), childHops);

            // Todo: TRead의 Child가 없는 경우 예외 처리 (왜 없는 지 확인)
            if (childHops == null || childHops.isEmpty()) {
                System.out.println("[RewireTransHop] (hopName: " + hopName + ", hopID: " + hop.getHopID() + ") child hops is empty");
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

            // Todo: TRead의 Filtered Child가 없는 경우 예외 처리 (왜 없는 지 확인)
            if (filteredChildHops.isEmpty()) {
                System.out.println("[RewireTransHop] (hopName: " + hopName + ", hopID: " + hop.getHopID() + ") filtered child hops is empty");
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
                    throw new DMLRuntimeException("TransRead의 입력 FType이 일치하지 않습니다. : " + inputFType + " != " + fTypeMap.get(filteredChildHopID));
                }
            }
            // Propagate Privacy Constraint
            privacyConstraintMap.put(hop.getHopID(),
                    getPrivacyConstraint(hop, filteredChildHops, privacyConstraintMap));
            // Propagate FType
            fTypeMap.put(hop.getHopID(), inputFType);
            
            logEnhancedFType(hop, true, inputFType, ReasonCode.PROPAGATE_FROM_INPUT, fTypeMap);
        } else {
            privacyConstraintMap.put(hop.getHopID(),
                    getPrivacyConstraint(hop, hop.getInput(), privacyConstraintMap));
            if (allowsFederated(hop, fTypeMap)) {
                FType resultFType = getFType(hop, fTypeMap);
                fTypeMap.put(hop.getHopID(), resultFType);
                
                logEnhancedFType(hop, true, resultFType, ReasonCode.DATA_FEDERATED, fTypeMap);
            } else {
                fTypeMap.put(hop.getHopID(), null);
                
                logEnhancedFType(hop, false, null, ReasonCode.DISALLOWED_OP, fTypeMap);
            }
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
            // 마지막으로 삽입된 outerTransTable부터 역순으로 순회
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
                FederatedResponse response = future.get(); // Future에서 실제 응답을 가져옴

                if (response.isSuccessful()) {
                    Object[] responseData = response.getData();
                    String privacyConstraints = (String) responseData[0]; // 프라이버시 제약조건을 문자열로 캐스팅
                    String pcLower = privacyConstraints.trim().toLowerCase();
                    Privacy tempPrivacy = null;

                    // 입력 문자열에 따라 적절한 PrivacyConstraint 값으로 매핑
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
                        throw new DMLRuntimeException("잘못된 개인정보 제약조건: " + privacyConstraints +
                                ". 'PRIVATE', 'PRIVATE_AGGREGATE', 'PUBLIC' 중 하나여야 합니다.");
                    }

                    if (privacyConstraint == null) {
                        privacyConstraint = tempPrivacy;
                    } else {
                        if (privacyConstraint != tempPrivacy) {
                            throw new DMLRuntimeException("개인정보 제약조건이 일치하지 않습니다.");
                        }
                    }
                } else {
                    // 에러 처리
                    String errorMsg = response.getErrorMessage();
                    System.err.println("프라이버시 제약조건 요청 실패: " + errorMsg);
                }
            } catch (Exception e) {
                // 예외 처리
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
    
    // Enhanced logging functions
    private static void logStepState(String stage, Hop hop, Map<Long, FType> fTypeMap) {
        callStackDepth.set(callStackDepth.get() + 1);
        LocalDateTime timestamp = LocalDateTime.now();
        
        System.out.printf("[StepState] %s | HopID: %d | Stage: %s | Depth: %d | Timestamp: %s%n",
            timestamp.format(TIMESTAMP_FORMAT),
            hop.getHopID(),
            stage,
            callStackDepth.get(),
            timestamp.format(TIMESTAMP_FORMAT)
        );
        
        if (stage.endsWith("_END")) {
            callStackDepth.set(Math.max(0, callStackDepth.get() - 1));
        }
    }
    
    private static void logEnhancedFType(Hop hop, boolean allowsFederated, FType resultFType, 
            ReasonCode reasonCode, Map<Long, FType> fTypeMap) {
        
        LocalDateTime timestamp = LocalDateTime.now();
        String hopName = hop.getName() != null ? hop.getName() : "unnamed";
        String hopType = hop.getClass().getSimpleName();
        String opCode = getOpCode(hop);
        
        // Collect input metadata
        String[] inputFTypes = new String[hop.getInput().size()];
        long[] inputHopIDs = new long[hop.getInput().size()];
        String[] inputNames = new String[hop.getInput().size()];
        
        for (int i = 0; i < hop.getInput().size(); i++) {
            Hop inputHop = hop.getInput(i);
            inputFTypes[i] = String.valueOf(fTypeMap.get(inputHop.getHopID()));
            inputHopIDs[i] = inputHop.getHopID();
            inputNames[i] = inputHop.getName() != null ? inputHop.getName() : "unnamed";
        }
        
        // Collect dimensions and sparsity
        long[] dimensions = {hop.getDim1(), hop.getDim2()};
        boolean isSparse = hop.getDataType() == DataType.MATRIX && hop.getNnz() > 0 && 
                           hop.getNnz() < (hop.getDim1() * hop.getDim2() * 0.1);
        
        // Print enhanced log
        System.out.printf("[FType] %s | HopID: %d | Name: %s | Type: %s | OpCode: %s | " +
                          "Depth: %d | allowsFederated: %b | ResultFType: %s | ReasonCode: %s | " +
                          "InputFTypes: %s | Dimensions: [%d,%d] | IsSparse: %b%n",
            timestamp.format(TIMESTAMP_FORMAT),
            hop.getHopID(),
            hopName,
            hopType,
            opCode,
            callStackDepth.get(),
            allowsFederated,
            resultFType,
            reasonCode,
            formatInputMetadata(inputHopIDs, inputNames, inputFTypes),
            dimensions[0],
            dimensions[1],
            isSparse
        );
    }
    
    private static void logDecisionPath(Hop hop, String[] conditions, String selectedBranch, 
            int alternativePaths, ReasonCode reasonCode) {
        
        LocalDateTime timestamp = LocalDateTime.now();
        
        System.out.printf("[DecisionPath] %s | HopID: %d | Conditions: %s | " +
                          "SelectedBranch: %s | AlternativePaths: %d | ReasonCode: %s%n",
            timestamp.format(TIMESTAMP_FORMAT),
            hop.getHopID(),
            String.join(", ", conditions),
            selectedBranch,
            alternativePaths,
            reasonCode
        );
    }
    
    private static String getOpCode(Hop hop) {
        if (hop instanceof AggUnaryOp) {
            return ((AggUnaryOp) hop).getOp().toString();
        } else if (hop instanceof AggBinaryOp) {
            return "AGGBINARY";
        } else if (hop instanceof UnaryOp) {
            return ((UnaryOp) hop).getOp().toString();
        } else if (hop instanceof BinaryOp) {
            return ((BinaryOp) hop).getOp().toString();
        } else if (hop instanceof TernaryOp) {
            return ((TernaryOp) hop).getOp().toString();
        } else if (hop instanceof ReorgOp) {
            return ((ReorgOp) hop).getOp().toString();
        } else if (hop instanceof DataOp) {
            return ((DataOp) hop).getOp().toString();
        } else if (hop instanceof FunctionOp) {
            return ((FunctionOp) hop).getFunctionName();
        } else if (hop instanceof NaryOp) {
            return ((NaryOp) hop).getOp().toString();
        } else if (hop instanceof ParameterizedBuiltinOp) {
            return ((ParameterizedBuiltinOp) hop).getOp().toString();
        } else if (hop instanceof DataGenOp) {
            return ((DataGenOp) hop).getOp().toString();
        } else if (hop instanceof DnnOp) {
            return "DNN";
        } else {
            return "UNKNOWN";
        }
    }
    
    private static String formatInputMetadata(long[] hopIDs, String[] names, String[] fTypes) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < hopIDs.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append(String.format("(hopID:%d,name:%s,ftype:%s)", hopIDs[i], names[i], fTypes[i]));
        }
        sb.append("]");
        return sb.toString();
    }
    
     private static boolean allowsFederated(Hop hop, Map<Long, FType> fTypeMap) {
	 	//generically obtain the input FTypes
	 	FType[] ft = new FType[hop.getInput().size()];

	 	for( int i=0; i<hop.getInput().size(); i++ )
	 		ft[i] = fTypeMap.get(hop.getInput(i).getHopID());

	 	// Initialize result variable and debugging info
	 	boolean result = false;
	 	ReasonCode reasonCode = ReasonCode.UNKNOWN_HOP_TYPE;
	 	List<String> conditions = new ArrayList<>();
	 	String selectedBranch = "UNKNOWN";
	 	int alternativePaths = 0;
	 	StringBuilder reason = new StringBuilder(); // Temporary for backward compatibility

	 	// AggUnaryOp operations
	 	if(hop instanceof AggUnaryOp && ft.length==1 && ft[0] != null) {
	 		AggOp aggOp = ((AggUnaryOp)hop).getOp();
	 		result = aggOp == AggOp.SUM || aggOp == AggOp.MIN || aggOp == AggOp.MAX;
	 		reasonCode = result ? ReasonCode.AGGR_UNARY_ALLOWED : ReasonCode.AGGR_UNARY_DISALLOWED;
	 		selectedBranch = "AGGR_UNARY_OP";
	 		alternativePaths = 1;
	 		conditions.add("ft.length==1:" + (ft.length == 1));
	 		conditions.add("ft[0]!=null:" + (ft[0] != null));
	 		conditions.add("op in [SUM,MIN,MAX]:" + result);
	 	}
	 	// AggBinaryOp operations
	 	else if( hop instanceof AggBinaryOp ) {
	 		boolean mixedNull = (ft[0] != null && ft[1] == null) || (ft[0] == null && ft[1] != null);
	 		boolean colRowPattern = (ft[0] == FType.COL && ft[1] == FType.ROW);
	 		result = mixedNull || colRowPattern;
			
	 		if (mixedNull) {
	 			reasonCode = ReasonCode.AGGR_BINARY_MIXED_NULL;
	 			selectedBranch = "MIXED_NULL_CASE";
	 		} else if (colRowPattern) {
	 			reasonCode = ReasonCode.AGGR_BINARY_COL_ROW;
	 			selectedBranch = "COL_ROW_CASE";
	 		} else {
	 			reasonCode = ReasonCode.AGGR_BINARY_PROPAGATE;
	 			selectedBranch = "NO_MATCH_CASE";
	 		}
			
	 		alternativePaths = 3;
	 		conditions.add("ft[0]!=null && ft[1]==null:" + (ft[0] != null && ft[1] == null));
	 		conditions.add("ft[0]==null && ft[1]!=null:" + (ft[0] == null && ft[1] != null));
	 		conditions.add("ft[0]==COL && ft[1]==ROW:" + colRowPattern);
	 	}
	 	// UnaryOp operations
	 	else if (hop instanceof UnaryOp) {
	 		UnaryOp uop = (UnaryOp) hop;
	 		OpOp1 op = uop.getOp();
	 		boolean isDisallowedOp = op == OpOp1.PRINT || op == OpOp1.ASSERT || op == OpOp1.STOP
	 			|| op == OpOp1.TYPEOF || op == OpOp1.INVERSE || op == OpOp1.EIGEN
	 			|| op == OpOp1.CHOLESKY || op == OpOp1.DET || op == OpOp1.SVD
	 			|| op == OpOp1.SQRT_MATRIX_JAVA || op == OpOp1.LOG || op == OpOp1.ROUND;
	 		boolean isListInput = hop.getInput().get(0).getDataType() == DataType.LIST;
	 		boolean isMetadata = uop.isMetadataOperation();
			
	 		result = !(isDisallowedOp || isListInput || isMetadata);
			
	 		if (isDisallowedOp) {
	 			reasonCode = ReasonCode.UNARY_DISALLOWED_OP;
	 			selectedBranch = "DISALLOWED_OP";
	 		} else if (isListInput) {
	 			reasonCode = ReasonCode.UNARY_LIST_INPUT;
	 			selectedBranch = "LIST_INPUT";
	 		} else if (isMetadata) {
	 			reasonCode = ReasonCode.UNARY_METADATA_OP;
	 			selectedBranch = "METADATA_OP";
	 		} else {
	 			reasonCode = ReasonCode.UNARY_ALLOWED;
	 			selectedBranch = "ALLOWED";
	 		}
			
	 		alternativePaths = 4;
	 		conditions.add("disallowed_op:" + isDisallowedOp);
	 		conditions.add("list_input:" + isListInput);
	 		conditions.add("metadata_op:" + isMetadata);
	 	}
	 	// BinaryOp operations (non-scalar)
	 	else if( hop instanceof BinaryOp && !hop.getDataType().isScalar() ) {
	 		OpOp2 op = ((BinaryOp) hop).getOp();
	 		if (op == OpOp2.MIN) {
	 			result = false;
	 			reasonCode = ReasonCode.BINARY_MIN_DISALLOWED;
	 			selectedBranch = "MIN_DISALLOWED";
	 			alternativePaths = 2;
	 			conditions.add("op==MIN:true");
	 		} else {
	 			boolean mixedNull = (ft[0] != null && ft[1] == null) || (ft[0] == null && ft[1] != null);
	 			boolean sameFType = (ft[0] != null && ft[0] == ft[1]);
	 			result = mixedNull || sameFType;
				
	 			if (mixedNull) {
	 				reasonCode = ReasonCode.BINARY_MIXED_NULL;
	 				selectedBranch = "MIXED_NULL";
	 			} else if (sameFType) {
	 				reasonCode = ReasonCode.BINARY_SAME_FTYPE;
	 				selectedBranch = "SAME_FTYPE";
	 			} else {
	 				reasonCode = ReasonCode.DISALLOWED_OP;
	 				selectedBranch = "NO_MATCH";
	 			}
				
	 			alternativePaths = 3;
	 			conditions.add("mixed_null:" + mixedNull);
	 			conditions.add("same_ftype:" + sameFType);
	 		}
	 	}
	 	// TernaryOp operations (non-scalar)
	 	else if( hop instanceof TernaryOp && !hop.getDataType().isScalar() ) {
	 		OpOp3 op = ((TernaryOp) hop).getOp();
	 		if (op == OpOp3.CTABLE || op == OpOp3.IFELSE) {
	 			result = false;
	 			reason.append("TernaryOp with operation: ").append(op)
	 				  .append(" (CTABLE/IFELSE always disallowed), result: false");
	 		} else {
	 			result = (ft[0] != null || ft[1] != null || ft[2] != null);
	 			reason.append("TernaryOp with operation: ").append(op)
	 				  .append(", ft[0]=").append(ft[0]).append(", ft[1]=").append(ft[1]).append(", ft[2]=").append(ft[2])
	 				  .append(", condition: at least one non-null FType")
	 				  .append(", result: ").append(result);
	 		}
	 	}
	 	// ReorgOp operations
	 	else if ( hop instanceof ReorgOp && ((ReorgOp)hop).getOp() == ReOrgOp.TRANS ){
	 		result = ft[0] == FType.COL || ft[0] == FType.ROW;
	 		reason.append("ReorgOp TRANS with ft[0]=").append(ft[0])
	 			  .append(", condition: ft[0] is COL or ROW")
	 			  .append(", result: ").append(result);
	 	}
	 	// DataOp operations
	 	else if (hop instanceof DataOp) {
	 		OpOpData op = ((DataOp) hop).getOp();
	 		result = op == OpOpData.FEDERATED
	 			|| op == OpOpData.TRANSIENTWRITE
	 			|| op == OpOpData.TRANSIENTREAD;
	 		reason.append("DataOp with operation: ").append(op)
	 			  .append(", allowed: [FEDERATED, TRANSIENTWRITE, TRANSIENTREAD]")
	 			  .append(", result: ").append(result);
	 	}
	 	// FunctionOp operations
	 	else if (hop instanceof FunctionOp) {
	 		FunctionOp fop = (FunctionOp) hop;
	 		String funcName = fop.getFunctionName();
	 		result = !funcName.equalsIgnoreCase(Opcodes.TRANSFORMENCODE.toString());
	 		reason.append("FunctionOp with name: ").append(funcName)
	 			  .append(", disallowed: TRANSFORMENCODE")
	 			  .append(", result: ").append(result);
	 	}
	 	// NaryOp operations
	 	else if (hop instanceof NaryOp) {
	 		OpOpN op = ((NaryOp) hop).getOp();
	 		boolean isDisallowedOp = op == OpOpN.PRINTF || op == OpOpN.EVAL || op == OpOpN.LIST;
	 		boolean isListCbind = op == OpOpN.CBIND && hop.getInput().get(0).getDataType().isList();
	 		boolean isListRbind = op == OpOpN.RBIND && hop.getInput().get(0).getDataType().isList();
			
	 		result = !(isDisallowedOp || isListCbind || isListRbind);
	 		reason.append("NaryOp with operation: ").append(op)
	 			  .append(", disallowed_op: ").append(isDisallowedOp)
	 			  .append(", list_cbind: ").append(isListCbind)
	 			  .append(", list_rbind: ").append(isListRbind)
	 			  .append(", result: ").append(result);
	 	}
	 	// ParameterizedBuiltinOp operations
	 	else if (hop instanceof ParameterizedBuiltinOp) {
	 		ParamBuiltinOp op = ((ParameterizedBuiltinOp) hop).getOp();
	 		result = !(op == ParamBuiltinOp.TOSTRING || op == ParamBuiltinOp.LIST
	 			|| op == ParamBuiltinOp.CDF || op == ParamBuiltinOp.INVCDF
	 			|| op == ParamBuiltinOp.PARAMSERV || op == ParamBuiltinOp.REXPAND
	 			|| op == ParamBuiltinOp.REPLACE);
	 		reason.append("ParameterizedBuiltinOp with operation: ").append(op)
	 			  .append(", disallowed: [TOSTRING, LIST, CDF, INVCDF, PARAMSERV, REXPAND, REPLACE]")
	 			  .append(", result: ").append(result);
	 	}
	 	// DataGenOp operations
	 	else if (hop instanceof DataGenOp) {
	 		OpOpDG op = ((DataGenOp) hop).getOp();
	 		result = !(op == OpOpDG.TIME || op == OpOpDG.SINIT || op == OpOpDG.RAND || op == OpOpDG.SEQ);
	 		reason.append("DataGenOp with operation: ").append(op)
	 			  .append(", disallowed: [TIME, SINIT, RAND, SEQ]")
	 			  .append(", result: ").append(result);
	 	}
	 	// DnnOp operations
	 	else if (hop instanceof DnnOp) {
	 		result = false;
	 		reason.append("DnnOp (always disallowed), result: false");
	 	}
	 	// Default case
	 	else {
	 		result = false;
	 		reason.append("Unknown hop type or no matching condition, result: false");
	 	}

	 	// Log decision path and result
	 	logDecisionPath(hop, conditions.toArray(new String[0]), selectedBranch, alternativePaths, reasonCode);
	 	logEnhancedFType(hop, result, null, reasonCode, fTypeMap);
		
	 	return result;
	 }

//	private static boolean allowsFederated(Hop hop, Map<Long, FType> fTypeMap) {
//		//generically obtain the input FTypes
//		FType[] ft = new FType[hop.getInput().size()];
//
//		for( int i=0; i<hop.getInput().size(); i++ )
//			ft[i] = fTypeMap.get(hop.getInput(i).getHopID());
//
//		// AggBinaryOp operations
//		if( hop instanceof AggBinaryOp ) {
//			return (ft[0] != null && ft[1] == null)
//				|| (ft[0] == null && ft[1] != null)
//				|| (ft[0] == FType.COL && ft[1] == FType.ROW);
//		}
//		// AggUnaryOp operations
//		else if(hop instanceof AggUnaryOp && ft.length==1 && ft[0] != null) {
//			AggOp aggOp = ((AggUnaryOp)hop).getOp();
//			return aggOp == AggOp.SUM || aggOp == AggOp.MIN || aggOp == AggOp.MAX;
//		}
//		// BinaryOp operations (non-scalar)
//		else if( hop instanceof BinaryOp && !hop.getDataType().isScalar() ) {
//			OpOp2 op = ((BinaryOp) hop).getOp();
//			if (op == OpOp2.MIN) {
//				return false;
//			}
//			return (ft[0] != null && ft[1] == null)
//				|| (ft[0] == null && ft[1] != null)
//				|| (ft[0] != null && ft[0] == ft[1]);
//		}
//		// DataGenOp operations
//		else if (hop instanceof DataGenOp) {
//			OpOpDG op = ((DataGenOp) hop).getOp();
//			return !(op == OpOpDG.TIME || op == OpOpDG.SINIT || op == OpOpDG.RAND || op == OpOpDG.SEQ);
//		}
//		// DataOp operations
//		else if (hop instanceof DataOp) {
//			OpOpData op = ((DataOp) hop).getOp();
//			return op == OpOpData.FEDERATED
//				|| op == OpOpData.TRANSIENTWRITE
//				|| op == OpOpData.TRANSIENTREAD;
//		}
//		// DnnOp operations
//		else if (hop instanceof DnnOp) {
//			return false;
//		}
//		// FunctionOp operations
//		else if (hop instanceof FunctionOp) {
//			FunctionOp fop = (FunctionOp) hop;
//			return !fop.getFunctionName().equalsIgnoreCase(Opcodes.TRANSFORMENCODE.toString());
//		}
//		// NaryOp operations
//		else if (hop instanceof NaryOp) {
//			OpOpN op = ((NaryOp) hop).getOp();
//			return !(op == OpOpN.PRINTF || op == OpOpN.EVAL || op == OpOpN.LIST
//				// cbind/rbind of lists only support in CP right now
//				|| (op == OpOpN.CBIND && hop.getInput().get(0).getDataType().isList())
//				|| (op == OpOpN.RBIND && hop.getInput().get(0).getDataType().isList()));
//		}
//		// ParameterizedBuiltinOp operations
//		else if (hop instanceof ParameterizedBuiltinOp) {
//			ParamBuiltinOp op = ((ParameterizedBuiltinOp) hop).getOp();
//			return !(op == ParamBuiltinOp.TOSTRING || op == ParamBuiltinOp.LIST
//				|| op == ParamBuiltinOp.CDF || op == ParamBuiltinOp.INVCDF
//				|| op == ParamBuiltinOp.PARAMSERV || op == ParamBuiltinOp.REXPAND
//				|| op == ParamBuiltinOp.REPLACE);
//		}
//		// ReorgOp operations
//		else if ( hop instanceof ReorgOp && ((ReorgOp)hop).getOp() == ReOrgOp.TRANS ){
//			return ft[0] == FType.COL || ft[0] == FType.ROW;
//		}
//		// TernaryOp operations (non-scalar)
//		else if( hop instanceof TernaryOp && !hop.getDataType().isScalar() ) {
//			OpOp3 op = ((TernaryOp) hop).getOp();
//			if (op == OpOp3.CTABLE || op == OpOp3.IFELSE) {
//				return false;
//			}
//			return (ft[0] != null || ft[1] != null || ft[2] != null);
//		}
//		// UnaryOp operations
//		else if (hop instanceof UnaryOp) {
//			UnaryOp uop = (UnaryOp) hop;
//			OpOp1 op = uop.getOp();
//			return !(op == OpOp1.PRINT || op == OpOp1.ASSERT || op == OpOp1.STOP
//				|| op == OpOp1.TYPEOF || op == OpOp1.INVERSE || op == OpOp1.EIGEN
//				|| op == OpOp1.CHOLESKY || op == OpOp1.DET || op == OpOp1.SVD
//				|| op == OpOp1.SQRT_MATRIX_JAVA || op == OpOp1.LOG || op == OpOp1.ROUND
//				|| hop.getInput().get(0).getDataType() == DataType.LIST
//				|| uop.isMetadataOperation());
//		}
//		return false;
//	}

     private static FType getFType(Hop hop, Map<Long, FType> fTypeMap){
         //generically obtain the input FTypes
         FType[] ft = new FType[hop.getInput().size()];
         for( int i=0; i<hop.getInput().size(); i++ )
             ft[i] = fTypeMap.get(hop.getInput(i).getHopID());
  
         // Initialize result variable and debugging info
         FType result = null;
         StringBuilder reason = new StringBuilder();
        
         // Collect basic hop information
         String hopType = hop.getClass().getSimpleName();
         String hopName = hop.getName() != null ? hop.getName() : "unnamed";
         long hopID = hop.getHopID();
        
         // Format input FTypes for logging
         StringBuilder ftInfo = new StringBuilder("[");
         for (int i = 0; i < ft.length; i++) {
             ftInfo.append(ft[i]);
             if (i < ft.length - 1) ftInfo.append(", ");
         }
         ftInfo.append("]");

         if ( hop.isScalar() ) {
             result = null;
             reason.append("Scalar hop, always returns null");
         }
	 	else if( hop instanceof AggBinaryOp ) {
	 		MMTSJType mmtsj = ((AggBinaryOp) hop).checkTransposeSelf() ; //determine tsmm pattern
	 		if ( mmtsj != MMTSJType.NONE &&
	 			(( mmtsj.isLeft() && ft[0] == FType.ROW ) || ( mmtsj.isRight() && ft[0] == FType.COL ) )) {
	 			result = FType.BROADCAST;
	 			reason.append("AggBinaryOp with TSMM pattern: ").append(mmtsj)
	 				  .append(", ft[0]=").append(ft[0])
	 				  .append(", condition: (isLeft && ft[0]==ROW) || (isRight && ft[0]==COL)")
	 				  .append(", result: BROADCAST");
	 		}
	 		else if( ft[0] != null ) {
	 			result = ft[0] == FType.ROW ? FType.ROW : null;
	 			reason.append("AggBinaryOp with ft[0]=").append(ft[0])
	 				  .append(", condition: ft[0]==ROW ? ROW : null")
	 				  .append(", result: ").append(result);
	 		}
	 		else {
	 			result = null;
	 			reason.append("AggBinaryOp with ft[0]=null, result: null");
	 		}
	 	}
	 	else if( hop instanceof BinaryOp ) {
	 		result = ft[0] != null ? ft[0] : ft[1];
	 		reason.append("BinaryOp with ft[0]=").append(ft[0]).append(", ft[1]=").append(ft[1])
	 			  .append(", condition: ft[0] != null ? ft[0] : ft[1]")
	 			  .append(", result: ").append(result);
	 	}
	 	else if( hop instanceof TernaryOp ) {
	 		result = ft[0] != null ? ft[0] : ft[1] != null ? ft[1] : ft[2];
	 		reason.append("TernaryOp with ft[0]=").append(ft[0]).append(", ft[1]=").append(ft[1]).append(", ft[2]=").append(ft[2])
	 			  .append(", condition: first non-null FType")
	 			  .append(", result: ").append(result);
	 	}
	 	else if( HopRewriteUtils.isReorg(hop, ReOrgOp.TRANS) ){
	 		if (ft[0] == FType.ROW) {
	 			result = FType.COL;
	 			reason.append("ReorgOp TRANS with ft[0]=ROW, result: COL");
	 		}
	 		else if (ft[0] == FType.COL) {
	 			result = FType.ROW;
	 			reason.append("ReorgOp TRANS with ft[0]=COL, result: ROW");
	 		}
	 		else {
	 			result = null;
	 			reason.append("ReorgOp TRANS with ft[0]=").append(ft[0]).append(", result: null");
	 		}
	 	}
	 	else if ( hop instanceof AggUnaryOp ){
	 		boolean isColAgg = ((AggUnaryOp) hop).getDirection().isCol();
	 		if ( (ft[0] == FType.ROW && isColAgg) || (ft[0] == FType.COL && !isColAgg) ) {
	 			result = null;
	 			reason.append("AggUnaryOp with ft[0]=").append(ft[0])
	 				  .append(", isColAgg=").append(isColAgg)
	 				  .append(", condition: incompatible aggregation direction")
	 				  .append(", result: null");
	 		}
	 		else if (ft[0] == FType.ROW || ft[0] == FType.COL) {
	 			result = ft[0];
	 			reason.append("AggUnaryOp with ft[0]=").append(ft[0])
	 				  .append(", isColAgg=").append(isColAgg)
	 				  .append(", condition: compatible aggregation direction")
	 				  .append(", result: ").append(result);
	 		}
	 		else {
	 			result = null;
	 			reason.append("AggUnaryOp with ft[0]=").append(ft[0])
	 				  .append(", isColAgg=").append(isColAgg)
	 				  .append(", result: null");
	 		}
	 	}
	 	else if ( HopRewriteUtils.isData(hop, Types.OpOpData.FEDERATED) ) {
	 		result = deriveFType((DataOp)hop);
	 		reason.append("DataOp FEDERATED, calling deriveFType")
	 			  .append(", result: ").append(result);
	 	}
	 	else if ( HopRewriteUtils.isData(hop, Types.OpOpData.TRANSIENTWRITE)
	 		|| HopRewriteUtils.isData(hop, Types.OpOpData.TRANSIENTREAD) ) {
	 		result = ft[0];
	 		String opType = HopRewriteUtils.isData(hop, Types.OpOpData.TRANSIENTWRITE) ? "TRANSIENTWRITE" : "TRANSIENTREAD";
	 		reason.append("DataOp ").append(opType)
	 			  .append(" with ft[0]=").append(ft[0])
	 			  .append(", propagating input FType")
	 			  .append(", result: ").append(result);
	 	}
	 	else {
	 		result = null;
	 		reason.append("Unknown hop type or no matching condition, result: null");
	 	}

	 	// Enhanced logging for getFType
	 	ReasonCode getFTypeReasonCode = ReasonCode.UNKNOWN_HOP_TYPE;
	 	if (hop.isScalar()) {
	 		getFTypeReasonCode = ReasonCode.SCALAR_HOP;
	 	} else if (hop instanceof AggBinaryOp) {
	 		getFTypeReasonCode = ReasonCode.TSMM_PATTERN;
	 	} else if (hop instanceof BinaryOp) {
	 		getFTypeReasonCode = ReasonCode.FIRST_NON_NULL;
	 	} else if (hop instanceof TernaryOp) {
	 		getFTypeReasonCode = ReasonCode.FIRST_NON_NULL;
	 	} else if (HopRewriteUtils.isReorg(hop, ReOrgOp.TRANS)) {
	 		getFTypeReasonCode = ReasonCode.REORG_TRANS_COL_ROW;
	 	} else if (hop instanceof AggUnaryOp) {
	 		getFTypeReasonCode = ReasonCode.AGGR_DIRECTION_MISMATCH;
	 	} else if (HopRewriteUtils.isData(hop, Types.OpOpData.FEDERATED)) {
	 		getFTypeReasonCode = ReasonCode.DERIVED_FROM_FED_RANGES;
	 	} else if (HopRewriteUtils.isData(hop, Types.OpOpData.TRANSIENTWRITE) ||
	 			   HopRewriteUtils.isData(hop, Types.OpOpData.TRANSIENTREAD)) {
	 		getFTypeReasonCode = ReasonCode.PROPAGATE_FROM_INPUT;
	 	}
		
	 	logEnhancedFType(hop, true, result, getFTypeReasonCode, fTypeMap);
		
	 	return result;
	 }

//    private static FType getFType(Hop hop, Map<Long, FType> fTypeMap){
//        //generically obtain the input FTypes
//        FType[] ft = new FType[hop.getInput().size()];
//        for( int i=0; i<hop.getInput().size(); i++ )
//            ft[i] = fTypeMap.get(hop.getInput(i).getHopID());
//
//        if ( hop.isScalar() )
//			return null;
//		if( hop instanceof AggBinaryOp ) {
//			MMTSJType mmtsj = ((AggBinaryOp) hop).checkTransposeSelf() ; //determine tsmm pattern
//			FType ret = null;
//
//            if ( mmtsj != MMTSJType.NONE &&
//				(( mmtsj.isLeft() && ft[0] == FType.ROW ) || ( mmtsj.isRight() && ft[0] == FType.COL ) ))
//				ret =  FType.BROADCAST;
//			else if( ft[0] != null )
//				ret = ft[0] == FType.ROW ? FType.ROW : null;
//            		//apply operator-specific heuristics
//			if( (ret == FType.ROW && hop.getDim2()==1) || (ret == FType.COL && hop.getDim1()==1) )
//				ret = null; //get local vectors
//			return ret;
//		}
//		else if ( hop instanceof AggUnaryOp ){
//			boolean isColAgg = ((AggUnaryOp) hop).getDirection().isCol();
//			if ( (ft[0] == FType.ROW && isColAgg) || (ft[0] == FType.COL && !isColAgg) )
//				return null;
//			else if (ft[0] == FType.ROW || ft[0] == FType.COL)
//				return ft[0];
//		}
//		else if( hop instanceof BinaryOp )
//			return ft[0] != null ? ft[0] : ft[1];
//		else if ( HopRewriteUtils.isData(hop, Types.OpOpData.FEDERATED) )
//			return deriveFType((DataOp)hop);
//		else if ( HopRewriteUtils.isData(hop, Types.OpOpData.TRANSIENTWRITE)
//			|| HopRewriteUtils.isData(hop, Types.OpOpData.TRANSIENTREAD) )
//			return ft[0];
//		else if( HopRewriteUtils.isReorg(hop, ReOrgOp.TRANS) ){
//			if (ft[0] == FType.ROW)
//				return FType.COL;
//			else if (ft[0] == FType.COL)
//				return FType.ROW;
//		}
//		else if( hop instanceof TernaryOp )
//			return ft[0] != null ? ft[0] : ft[1] != null ? ft[1] : ft[2];
//		return null;
//	}

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
        VariableSet genHops = sb.getGen();
        VariableSet updatedHops = sb.variablesUpdated();
        VariableSet liveOutHops = sb.liveOut();

        for (Long unRefTwriteHopID : unRefTwriteSet) {
            Hop unRefTwriteHop = hopCommonTable.get(unRefTwriteHopID).getHopRef();
            String unRefTwriteHopName = unRefTwriteHop.getName();

            if (liveOutHops.containsVariable(unRefTwriteHopName)) {
                continue;
            }

            if (genHops.containsVariable(unRefTwriteHopName) || updatedHops.containsVariable(unRefTwriteHopName)) {
                Iterator<String> liveOutHopsIterator = liveOutHops.getVariableNames().iterator();

                boolean isRewired = false;
                while (liveOutHopsIterator.hasNext()) {
                    String liveOutHopName = liveOutHopsIterator.next();
                    List<Hop> liveOutHopsList = newFormerTransTable.get(liveOutHopName);

                    if (liveOutHopsList != null && !liveOutHopsList.isEmpty()) {
                        List<Hop> copyLiveOutHopsList = new ArrayList<>(liveOutHopsList);
                        copyLiveOutHopsList.add(unRefTwriteHop);
                        newFormerTransTable.put(liveOutHopName, copyLiveOutHopsList);
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
