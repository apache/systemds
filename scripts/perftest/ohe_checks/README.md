<!--
{% comment %}
Licensed to the Apache Software Foundation (ASF) under one or more
contributor license agreements.  See the NOTICE file distributed with
this work for additional information regarding copyright ownership.
The ASF licenses this file to you under the Apache License, Version 2.0
(the "License"); you may not use this file except in compliance with
the License.  You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
{% end comment %}
-->

# Checking One Hot Encodedness before Compression tests

To run all tests for One Hot Encoding Checks:
 * install systemds,
 * make sure that the paths for SYSTEMDS_ROOT, JAVA_HOME, HADOOP_HOME, LOG4JPROP are correctly set
 * run experiments.sh

Alternatively, to run the experiment.dml script directly with OHE checks enabled, use this command: 
 
`$SYSTEMDS_ROOT/bin/systemds $SYSTEMDS_ROOT/target/SystemDS.jar experiment.dml --config ohe.xml `
 
Note: You can use -nvargs to set the variables rows, cols, dummy, distinct, repeats (how many times you want to generate a random matrix, transform-encode it and compress it)

(Dummy is the array of column indexes that you would like to One Hot Encode, example: dummy="[1]" will One Hot Encode the first column)

To collect the metrics from the logs for easier comparison, you can run `parse_logs.py` and an excel file called `combined_metrics.xlsx` will be created in this directory.
---
# Documentation of Changes to codebase for Implementing OHE Checks

## Flag to enable/disable OHE checks (Disabled by default)
- Added ``COMPRESSED_ONEHOTDETECT = "sysds.compressed.onehotdetect"`` to ``DMLConfig`` and adjusted the relevant methods
- Added attribute to ``CompressionSettings`` ``public final boolean oneHotDetect`` and adjusted the methods
- Adjusted ``CompressionSettingsBuilder`` to check if ``COMPRESSED_ONEHOTDETECT`` has been set to true to enable the checks 

## Changes in  `CoCoderFactory` 

### 1. Introduction of OHE Detection

**Condition Addition:**
- Added a condition to check for `cs.oneHotDetect` along with the existing condition `!containsEmptyConstOrIncompressable` in the `findCoCodesByPartitioning` method. This ensures that the process considers OHE detection only if it is enabled in the compression settings.
- Original code only checked for `containsEmptyConstOrIncompressable` and proceeded to cocode all columns if false. The updated code includes an additional check for `cs.oneHotDetect`.

### 2. New Data Structures for OHE Handling

**New Lists:** Introduced two new lists to manage the OHE detection process:
- `currentCandidates`: To store the current candidate columns that might form an OHE group.
- `oheGroups`: To store lists of columns that have been validated as OHE groups.

### 3. Filtering Logic Enhancements

**Column Filtering:** Enhanced the loop that iterates over columns to identify OHE candidates:
- Columns that are empty, constant, or incompressible are filtered into respective lists.
- For other columns, they are added to `currentCandidates` if they are deemed candidates (via `isCandidate` function).

### 4. Addition of `isHotEncoded` Function

**Function Creation:** Created a new `isHotEncoded` function to evaluate if the accumulated columns form a valid OHE group.
- **Parameters:** Takes a list of column groups (`colGroups`), a boolean flag (`isSample`), an array of non-zero counts (`nnzCols`), and the number of rows (`numRows`).
- **Return Type:** Returns a `String` indicating the status of the current candidates:
  - `"POTENTIAL_OHE"`: When the current candidates could still form an OHE group.
  - `"NOT_OHE"`: When the current candidates cannot form an OHE group.
  - `"VALID_OHE"`: When the current candidates form a valid OHE group.
- **Logic:** The function calculates the total number of distinct values and offsets, and checks if they meet the criteria for forming an OHE group.

### 5. Enhanced Group Handling

**Candidate Processing:** Within the loop, after adding a column to `currentCandidates`:
- Calls `isHotEncoded` to check the status of the candidates.
- If `isHotEncoded` returns `"NOT_OHE"`, moves the candidates to regular groups and clears the candidates list.
- If `isHotEncoded` returns `"VALID_OHE"`, moves the candidates to `oheGroups` and clears the candidates list.
- If `isHotEncoded` returns `"POTENTIAL_OHE"`, continues accumulating candidates.

### 6. Final Candidate Check

**Post-loop Check:** After the loop, checks any remaining `currentCandidates`:
- If they form a valid OHE group, adds them to `oheGroups`.
- Otherwise, adds them to regular groups.

### 7. Overwrite and CoCode Groups

**Overwrite Groups:** Updates `colInfos.compressionInfo` with the processed `groups`.
**OHE Group Integration:** Combines indexes for validated OHE groups and adds them to the final `groups`.

## One Hot Encoded Columns Compression in `ColGroupFactory`

### Description

The `compressOHE` function is designed to compress columns that are one-hot encoded (OHE). It validates and processes the input data to ensure it meets the criteria for one-hot encoding, and if so, it compresses the data accordingly. If the data does not meet the OHE criteria, it falls back to a direct compression method (`directCompressDDC`).

### Implementation Details

1. **Validation of `numVals`**:
   - Ensures the number of distinct values (`numVals`) in the column group is greater than 0.
   - Throws a `DMLCompressionException` if `numVals` is less than or equal to 0.

2. **Handling Transposed Matrix**:
   - If the matrix is transposed (`cs.transposed` is `true`):
     - Creates a `MapToFactory` data structure with an additional unique value.
     - Iterates through the sparse block of the matrix, checking for non-one values or multiple ones in the same row.
     - If a column index in the sparse block is empty, or if non-one values or multiple ones are found, it falls back to `directCompressDDC`.

3. **Handling Non-Transposed Matrix**:
   - If the matrix is not transposed (`cs.transposed` is `false`):
     - Creates a `MapToFactory` data structure.
     - Iterates through each row of the matrix:
       - Checks for the presence of exactly one '1' in the columns specified by `colIndexes`.
       - If multiple ones are found in the same row, or if no '1' is found in a sample row, it falls back to `directCompressDDC`.

4. **Return Value**:
   - If the data meets the OHE criteria, returns a `ColGroupDDC` created with the column indexes, an `IdentityDictionary`, and the data.
   - If the data does not meet the OHE criteria, returns the result of `directCompressDDC`.

## Add method in `ColGroupSizes`
Added method ``estimateInMemorySizeOHE(int nrColumns, boolean contiguousColumns, int nrRows)``

## Add method in `AComEst`
Added a getter method `getNnzCols`

## Edit `distinctCountScale` method in `ComEstSample`
```java 
if(freq == null || freq.length == 0)
    return numOffs+1;
```
And added condition:
```java
if(sampleFacts.numRows>sampleFacts.numOffs)
    est += 1;
```
<span style="color:red">Warning: This Change will cause some tests to fail</span>.


## Edit constructor in `CompressedSizeInfoColGroup`
Added a case in switch statement for OHE

## Added attribute in `CompressionStatistics`
Added Sparsity of input matrix attribute ``public double sparsity;`` to add logging in ``CompressedMatrixBlockFactory``
## Fix Bug in `extractFacts` method in `SparseEncoding`
Number of distinct values returned was wrong. 
Fix: In the return statements, changed map.getUnique() to getUnique()

## Fix Bug in `outputMatrixPostProcessing` method in `MultiColumnEncoder`
Instead of just recomputing nonzeroes in the else block, added `output.examSparsity(k);`
