# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Apache SystemDS — an end-to-end ML system that compiles DML scripts (R-like syntax) into hybrid execution plans across local (CP), Apache Spark, GPU, and federated backends.

## Build Commands

```bash
# Full build (skips tests)
mvn clean package -DskipTests

# Build and run default tests
mvn clean package

# Run a single Java test class
mvn test -Dtest=FullMatrixMultiplicationTest

# Run a test with a specific method
mvn test -Dtest=FullMatrixMultiplicationTest#testMethod

# Run checkstyle (disabled by default)
mvn checkstyle:check -Dcheckstyle.skip=false

# Python tests (from src/main/python/)
cd src/main/python && source python_venv/bin/activate && pip install -e . && python -m pytest tests/

# Run a DML script directly (after building)
./bin/systemds hello.dml
```

The surefire default test is `org.apache.sysds.test.usertest.**`. Override with `-Dtest=` to target specific classes. Test JVM is configured with `-Xmx3000m`.

## Code Style

Apply the Eclipse formatter profile at [dev/CodeStyle_eclipse.xml](dev/CodeStyle_eclipse.xml) before committing. Checkstyle rules are at [dev/checkstyle/checkstyle.xml](dev/checkstyle/checkstyle.xml).

## Commit Tags

All commits must be prefixed: `[SYSTEMDS-#]` for Jira issues, `[MINOR]` for small changes, `[DOC]` for docs, `[HOTFIX]` for release patches. The project uses linear history — rebase, never merge commits.

## Compilation Pipeline

The full compilation sequence is in [DMLScript.java:460-510](src/main/java/org/apache/sysds/api/DMLScript.java#L460):

1. **Parse** — DML source → `DMLProgram` AST (ANTLR-based, `parser/dml/`)
2. **Live Variable Analysis + Validate** — `DMLTranslator.liveVariableAnalysis()` / `validateParseTree()`
3. **Construct HOPs** — AST → HOP DAG (`DMLTranslator.constructHops()`)
4. **Rewrite HOP DAGs** — algebraic simplifications, IPA, memory estimates, CSE (`hops/rewrite/`)
5. **Construct LOPs** — HOP DAG → LOP DAG; exec type (CP/Spark/GPU/Fed) selected here
6. **Rewrite LOP DAGs** — inject prefetch, broadcast, OOC tee operators
7. **Generate runtime program** — LOPs emit instruction strings; codegen (Spoof) fuses operators
8. **Execute** — `ProgramBlock` hierarchy interprets instructions via `ExecutionContext`

## Architecture Patterns

### Two-Level IR: HOPs → LOPs
`Hop` ([hops/Hop.java](src/main/java/org/apache/sysds/hops/Hop.java)) is the algebraic IR. Each concrete subclass (`BinaryOp`, `AggBinaryOp`, `UnaryOp`, etc.) implements abstract template methods:
- `constructLops()` — lowers this operator to backend-specific LOPs
- `optFindExecType()` — selects CP / Spark / GPU / Fed based on cost model
- `inferOutputCharacteristics()` — size/sparsity estimation
- `computeOutputMemEstimate()` / `computeIntermediateMemEstimate()` — memory cost

`Lop` ([lops/Lop.java](src/main/java/org/apache/sysds/lops/Lop.java)) is the backend-specific IR. LOPs emit instruction strings consumed by the runtime.

### DAGs, not Trees
Both HOPs and LOPs are DAGs with explicit `_input`/`_parent` lists and a `_visited` boolean. All rewrite/lowering passes do DFS traversal with the visited flag. This enables CSE — shared subcomputations appear once in the DAG.

### Chain of Responsibility for Rewrites
`ProgramRewriter` ([hops/rewrite/ProgramRewriter.java](src/main/java/org/apache/sysds/hops/rewrite/ProgramRewriter.java)) holds an ordered list of `HopRewriteRule` subclasses and fires each over the full DAG. To add an optimization: subclass `HopRewriteRule`, implement `rewriteHopDAGs()` / `rewriteHopDAG()`, and register it. Existing rules include loop vectorization, loop-invariant hoisting, Spark checkpoint injection, OOC tee injection, and compressed reblock.

### Parallel AST ↔ Runtime Hierarchies
The parser (`Statement`/`StatementBlock`) and runtime (`ProgramBlock`) mirror each other exactly: `ForStatement` → `ForProgramBlock`, `ParForStatement` → `ParForProgramBlock`, etc. The AST is structural only; all execution behavior lives in the runtime mirror.

### String-Serialized Instruction Boundary
LOPs emit plain strings as instructions. At runtime, `InstructionParser.parseSingleInstruction()` reads the exec-type prefix (`CP·`, `SPARK·`, `GPU·`, `FED·`, `OOC·`) and dispatches to the appropriate backend parser. This decouples the compiler from all backend implementations.

### Caching Layer
`CacheableData<T>` ([runtime/controlprogram/caching/CacheableData.java](src/main/java/org/apache/sysds/runtime/controlprogram/caching/CacheableData.java)) is a generic abstract envelope for large data objects. Subclasses (`MatrixObject`, `FrameObject`, `TensorObject`) inherit full eviction/spill-to-disk lifecycle.

## Key Source Locations

| Area | Path |
|---|---|
| Compiler entry point | `src/main/java/org/apache/sysds/api/DMLScript.java` |
| DML → HOP translator | `src/main/java/org/apache/sysds/parser/DMLTranslator.java` |
| HOP base + optimizer | `src/main/java/org/apache/sysds/hops/` |
| HOP rewrites | `src/main/java/org/apache/sysds/hops/rewrite/` |
| LOP base | `src/main/java/org/apache/sysds/lops/` |
| Runtime control flow | `src/main/java/org/apache/sysds/runtime/controlprogram/` |
| Runtime instructions | `src/main/java/org/apache/sysds/runtime/instructions/` |
| Compressed Linear Algebra | `src/main/java/org/apache/sysds/runtime/compress/` |
| Operator fusion (Spoof) | `src/main/java/org/apache/sysds/runtime/codegen/` |
| Out-of-core execution | `src/main/java/org/apache/sysds/runtime/ooc/` |
| Built-in algorithms (DML) | `scripts/builtin/` |
| Staging area (new algos) | `scripts/staging/` |
| Python API | `src/main/python/systemds/` |

## Testing

Java tests extend `AutomatedTestBase`. The default exec mode is `ExecMode.HYBRID`; tests call `setExecMode()` / `resetExecMode()` to test specific backends.

- **`src/test/java/.../test/functions/`** — end-to-end integration tests that run DML scripts through the full pipeline
- **`src/test/java/.../test/component/`** — unit tests targeting specific subsystems (compress, matrix, codegen, ooc, parfor, etc.)
- **`src/test/java/.../test/applications/`** — full ML algorithm correctness tests

Each function test has a paired DML script under `src/test/scripts/functions/`.

## Adding a New Built-in Algorithm

New algorithms live in `scripts/staging/` until they have sufficient test coverage, then move to `scripts/builtin/`. The Python API wrappers in `src/main/python/systemds/operator/algorithm/builtin/` are auto-generated — see `src/main/python/generator/` for the generation scripts.
