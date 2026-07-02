# Claude Code Plan: Differential-Privacy Built-ins for Apache SystemDS

## Goal
Add `dp_laplace` and `dp_gaussian` as native (non-script) DML built-in
functions backed by a session-scoped Rényi-DP budget accountant.

This plan is written for Claude Code. Follow the steps in order. Read each
file before editing it. Do not guess at class names, field names, or method
signatures — grep to verify every assumption before writing code.

---

## Step 1 — Orient: understand the instruction routing mechanism

```bash
# 1a. Find how opcodes are mapped to CPInstruction subclasses.
#     We are looking for whatever replaces (or still is) the opcode→type map.
grep -rn "parseSingleInstruction\|CPType\|CPInstructionParser" \
  src/main/java/org/apache/sysds/runtime/instructions/CPInstructionParser.java \
  | head -60
```

```bash
# 1b. Look for how a recently-added native instruction (e.g. lstm, compress)
#     is wired in. This gives us the exact pattern to copy.
grep -n "lstm\|compress\|bias_add" \
  src/main/java/org/apache/sysds/runtime/instructions/CPInstructionParser.java \
  | head -30
```

```bash
# 1c. Read the full parseSingleInstruction method to understand the switch/map
#     dispatch that leads to parseInstruction() on a specific class.
grep -n "parseSingleInstruction\|case Dnn\|case Builtin\|DnnCPInstruction\
\|ParameterizedBuiltin\|parseInstruction" \
  src/main/java/org/apache/sysds/runtime/instructions/CPInstructionParser.java \
  | head -60
```

> **Note for Claude Code**: the exact field/method name may differ from
> `String2CPInstructionType`. Use the output of step 1b to find the real
> registration pattern. Copy it exactly — do not invent names.

---

## Step 2 — Orient: understand the Builtins enum constructor

```bash
# 2a. Read the constructor and the first ~60 enum entries to confirm the
#     exact parameter signature (name, script) vs (name, script, ReturnType).
sed -n '35,100p' \
  src/main/java/org/apache/sysds/common/Builtins.java
```

```bash
# 2b. Confirm no "parameterized" field exists in the constructor.
grep -n "parameterized\|Parameterized\|boolean" \
  src/main/java/org/apache/sysds/common/Builtins.java | head -20
```

> Expected: constructor is `(String name, boolean script)` with `script=false`
> for native built-ins. Verify before proceeding.

---

## Step 3 — Orient: understand BuiltinFunctionExpression validation

```bash
# 3a. Find how an existing similar native built-in (e.g. colMeans, abs)
#     is validated in the parser and how it maps to a HOP.
grep -n "COLMEAN\|ABS\|BuiltinFunctionExpression\|case COLMEAN\|case ABS" \
  src/main/java/org/apache/sysds/parser/BuiltinFunctionExpression.java \
  | head -30
```

```bash
# 3b. Find DMLTranslator to understand how BuiltinFunctionExpression
#     creates a HOP node.
grep -n "createBuiltin\|BuiltinOp\|UnaryOp\|colMeans\|case COLMEAN" \
  src/main/java/org/apache/sysds/parser/DMLTranslator.java \
  | head -20
```

```bash
# 3c. Find how the HOP emits a LOP that becomes a CPInstruction opcode string.
grep -n "getOpCode\|getLops\|addLop\|Lops" \
  src/main/java/org/apache/sysds/hops/UnaryOp.java \
  | head -20
```

> This tells us whether we need a new HOP type, a new LOP type, or whether
> dp_laplace/dp_gaussian can reuse an existing HOP+LOP path.

---

## Step 4 — Create the new files

Create the following files. All paths are relative to the repository root.

### 4a. RDPAccountant

**File**: `src/main/java/org/apache/sysds/runtime/privacy/dp/RDPAccountant.java`

Verify the package declaration matches the target path:
```bash
head -5 src/main/java/org/apache/sysds/runtime/privacy/dp/RDPAccountant.java
```

### 4b. DPBuiltinCPInstruction

**File**: `src/main/java/org/apache/sysds/runtime/instructions/cp/DPBuiltinCPInstruction.java`

Verify the imports compile against the actual codebase:

```bash
# Confirm BinaryOperator and Plus exist at the expected paths.
find src/main/java -name "BinaryOperator.java" -o -name "Plus.java" | head -5

# Confirm MatrixBlock.binaryOperations signature.
grep -n "binaryOperations" \
  src/main/java/org/apache/sysds/runtime/matrix/data/MatrixBlock.java \
  | head -10
```

If `MatrixBlock.binaryOperations` has a different signature, update the call
in `processInstruction` to match.

---

## Step 5 — Patch ExecutionContext to carry the accountant

```bash
# 5a. Read the end of ExecutionContext to find where to add the new field.
grep -n "class ExecutionContext\|private.*Map\|private.*List\|getMatrix\
\|releaseMatrix\|setScalar" \
  src/main/java/org/apache/sysds/runtime/controlprogram/context/ExecutionContext.java \
  | tail -40
```

Add the following to `ExecutionContext.java`:

```java
// --- DP budget accountant (lazy-initialised, one per script execution) ---
private RDPAccountant _rdpAccountant = null;

public RDPAccountant getRDPAccountant() {
    if (_rdpAccountant == null)
        // Default budget: ε=1.0, δ=1e-5. Future work: set via DML built-in.
        _rdpAccountant = new RDPAccountant(1.0, 1e-5);
    return _rdpAccountant;
}
```

Add the import at the top of `ExecutionContext.java`:
```java
import org.apache.sysds.runtime.privacy.dp.RDPAccountant;
```

---

## Step 6 — Register in Builtins.java

```bash
# 6a. Find a good alphabetical insertion point between "D" entries.
grep -n "^[[:space:]]*D[A-Z_]*(" \
  src/main/java/org/apache/sysds/common/Builtins.java | head -20
```

Insert after the last `D`-prefixed entry (or before the first `E` entry):

```java
DP_LAPLACE("dp_laplace", false),
DP_GAUSSIAN("dp_gaussian", false),
```

Confirm `script=false` is correct by checking a nearby native built-in:
```bash
grep -A1 "DIAG\|DECOMPRESS\|DET" \
  src/main/java/org/apache/sysds/common/Builtins.java | head -10
```

---

## Step 7 — Wire into the parser

Use the routing pattern discovered in Step 1. The pattern will be one of:

**Pattern A — opcode map + switch** (most likely based on commit history):
```bash
# Find the CPType enum to add a new entry if needed.
grep -n "enum CPType\|Dnn,\|BuiltinNary,\|ParameterizedBuiltin," \
  src/main/java/org/apache/sysds/runtime/instructions/cp/CPInstruction.java \
  | head -20
```

If a new `CPType.DPBuiltin` is needed, add it to the `CPType` enum, then add
the map entry and switch case following the exact pattern of `CPType.Dnn`.

**Pattern B — direct opcode string match in parseSingleInstruction**:
Add a branch:
```java
else if (opcode.equals("dp_laplace") || opcode.equals("dp_gaussian"))
    return DPBuiltinCPInstruction.parseInstruction(str);
```

> Use whichever pattern the codebase actually uses. Do not mix patterns.

---

## Step 8 — Wire into BuiltinFunctionExpression

```bash
# 8a. Find the validate() switch to add parameter checking.
grep -n "case DIAG\|case ABS\|case COLMEAN\|checkNumParameters\|checkMatrixParam" \
  src/main/java/org/apache/sysds/parser/BuiltinFunctionExpression.java \
  | head -20
```

Add cases for `DP_LAPLACE` and `DP_GAUSSIAN` in the `validate()` switch:

```java
case DP_LAPLACE: {
    // dp_laplace(aggregate, sensitivity, epsilon)
    checkNumParameters(3);
    checkMatrixParam(getFirstExpr());   // aggregate matrix
    checkScalarParam(getSecondExpr());  // sensitivity
    checkScalarParam(getThirdExpr());   // epsilon
    output.setDataType(DataType.MATRIX);
    output.setValueType(ValueType.FP64);
    // Output shape matches input shape; dimensions copied from input.
    output.setDimensions(
        getFirstExpr().getOutput().getDim1(),
        getFirstExpr().getOutput().getDim2());
    break;
}
case DP_GAUSSIAN: {
    // dp_gaussian(aggregate, sensitivity, epsilon, delta)
    checkNumParameters(4);
    checkMatrixParam(getFirstExpr());
    checkScalarParam(getSecondExpr());  // sensitivity
    checkScalarParam(getThirdExpr());   // epsilon
    checkScalarParam(getFourthExpr());  // delta
    output.setDataType(DataType.MATRIX);
    output.setValueType(ValueType.FP64);
    output.setDimensions(
        getFirstExpr().getOutput().getDim1(),
        getFirstExpr().getOutput().getDim2());
    break;
}
```

> Verify that `checkScalarParam` and `getFourthExpr` exist:
> ```bash
> grep -n "checkScalarParam\|getFourthExpr\|getThirdExpr" \
>   src/main/java/org/apache/sysds/parser/BuiltinFunctionExpression.java \
>   | head -10
> ```

---

## Step 9 — Wire into DMLTranslator

```bash
# 9a. Find how colMeans (COLMEAN) translates to a HOP in DMLTranslator.
grep -n "COLMEAN\|case COLMEAN\|createBuiltinOp\|UnaryOp" \
  src/main/java/org/apache/sysds/parser/DMLTranslator.java \
  | head -20
```

```bash
# 9b. Find where to add new cases — likely inside a large switch on Builtins.
grep -n "case ABS\|case DIAG\|case CEIL" \
  src/main/java/org/apache/sysds/parser/DMLTranslator.java | head -10
```

The simplest approach: map `DP_LAPLACE` and `DP_GAUSSIAN` to a `UnaryOp` HOP
with a custom opcode string. The LOP produced by `UnaryOp` will carry the
opcode string `"dp_laplace"` or `"dp_gaussian"`, which the `CPInstructionParser`
will then route to `DPBuiltinCPInstruction.parseInstruction`.

Add cases following the existing unary pattern:
```java
case DP_LAPLACE:
case DP_GAUSSIAN:
    // Reuse UnaryOp HOP — the opcode string routes to DPBuiltinCPInstruction.
    currBuiltinOp = new UnaryOp(target.getName(), DataType.MATRIX,
        ValueType.FP64, OpOp1.valueOf(bi.name()), expr);
    break;
```

> Verify `OpOp1` has a compatible entry or whether a different HOP class is
> needed. If `OpOp1` does not work, fall back to creating a `FunctionOp`.

---

## Step 10 — Build and smoke test

```bash
# Build only the affected modules to get fast feedback.
mvn compile -pl src/main/java -am -q 2>&1 | tail -30
```

Fix any compilation errors before proceeding.

```bash
# Run the self-contained unit tests (no SystemDS runtime needed).
mvn test -pl src/test/java \
  -Dtest=DPBuiltinCPInstructionTest \
  -Dsurefire.failIfNoSpecifiedTests=false \
  2>&1 | tail -40
```

```bash
# Smoke-test end-to-end with a minimal DML script.
cat > /tmp/dp_smoke.dml << 'EOF'
X = rand(rows=100, cols=10, min=0, max=1);
mu = colMeans(X);
noisy = dp_laplace(mu, sensitivity=0.1, epsilon=1.0);
print(toString(noisy));
EOF

./bin/systemds /tmp/dp_smoke.dml 2>&1 | tail -20
```

If the smoke test fails with an opcode-not-found error, re-check Steps 7 and 9.
If it fails with a budget error, reduce the sensitivity or widen epsilon.

---

## Step 11 — Run the federated benchmark

```bash
cat > /tmp/dp_fedavg_benchmark.dml << 'EOF'
# Federated averaging with DP release of column means.
# Sweep epsilon across {0.5, 1, 4, 8} by passing $epsilon as an arg.
X     = read($1);
mu    = colMeans(X);
noisy = dp_gaussian(mu, sensitivity=$sensitivity, epsilon=$epsilon, delta=1e-5);
write(noisy, $2, format="csv");
EOF

for eps in 0.5 1 4 8; do
  echo "--- epsilon=$eps ---"
  ./bin/systemds /tmp/dp_fedavg_benchmark.dml \
    -args data/adult.csv /tmp/result_eps${eps}.csv \
    -nvargs sensitivity=0.01 epsilon=$eps \
    2>&1 | tail -5
done
```

---

## Files modified (summary)

| File | Change |
|---|---|
| `src/main/java/org/apache/sysds/common/Builtins.java` | Add `DP_LAPLACE`, `DP_GAUSSIAN` entries |
| `src/main/java/org/apache/sysds/parser/BuiltinFunctionExpression.java` | Add validate cases |
| `src/main/java/org/apache/sysds/parser/DMLTranslator.java` | Add HOP-creation cases |
| `src/main/java/org/apache/sysds/runtime/instructions/CPInstructionParser.java` | Register opcodes using actual routing pattern |
| `src/main/java/org/apache/sysds/runtime/instructions/cp/DPBuiltinCPInstruction.java` | **New file** |
| `src/main/java/org/apache/sysds/runtime/privacy/dp/RDPAccountant.java` | **New file** |
| `src/main/java/org/apache/sysds/runtime/controlprogram/context/ExecutionContext.java` | Add `getRDPAccountant()` |
| `src/test/java/org/apache/sysds/test/functions/privacy/dp/DPBuiltinCPInstructionTest.java` | **New file** (unit tests) |
