# -------------------------------------------------------------
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
# -------------------------------------------------------------
#
# Unit tests for the pure logic in dev/format_changed.py (the line-scoping math,
# hunk parsing, and exclude matching). Run with: python -m pytest dev/tests
#
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import format_changed as fc  # noqa: E402


# --- overlaps: replace/delete regions vs edited ranges -------------------------

def test_overlaps_replace_region_hits_and_misses():
	# replace of original lines 5..7 -> 0-based half-open [4, 7)
	assert fc.overlaps(4, 7, [(6, 6)]) is True    # inside
	assert fc.overlaps(4, 7, [(5, 5)]) is True    # first line of region
	assert fc.overlaps(4, 7, [(7, 7)]) is True    # last line of region
	assert fc.overlaps(4, 7, [(4, 4)]) is False   # one before
	assert fc.overlaps(4, 7, [(8, 8)]) is False   # one after
	assert fc.overlaps(4, 7, [(1, 2)]) is False   # entirely before
	assert fc.overlaps(4, 7, [(8, 9)]) is False   # entirely after


def test_overlaps_multiple_ranges():
	assert fc.overlaps(4, 7, [(1, 2), (7, 9)]) is True
	assert fc.overlaps(4, 7, [(1, 2), (10, 11)]) is False


# --- overlaps: pure insertions (i1 == i2) -------------------------------------

def test_overlaps_pure_insertion_boundaries():
	# insertion at 0-based index 5 = between 1-based lines 5 and 6
	assert fc.overlaps(5, 5, [(5, 5)]) is True    # i1 == e
	assert fc.overlaps(5, 5, [(6, 6)]) is True    # i1 == s - 1
	assert fc.overlaps(5, 5, [(7, 7)]) is False   # below the range
	assert fc.overlaps(5, 5, [(3, 4)]) is False   # above the range
	assert fc.overlaps(0, 0, [(1, 1)]) is True    # insert before the first line


# --- _hunk_range / parse_hunks ------------------------------------------------

def test_hunk_range_multiline():
	assert fc._hunk_range("@@ -1,3 +10,4 @@") == (10, 13)


def test_hunk_range_missing_count_defaults_to_one():
	assert fc._hunk_range("@@ -0,0 +5 @@") == (5, 5)


def test_hunk_range_pure_deletion_is_none():
	assert fc._hunk_range("@@ -4,2 +3,0 @@") is None


def test_hunk_range_non_header_is_none():
	assert fc._hunk_range("+ some added line") is None
	assert fc._hunk_range("public int mul(int a) {") is None


def test_parse_hunks_collects_all_ranges():
	diff = (
		"diff --git a/X.java b/X.java\n"
		"--- a/X.java\n"
		"+++ b/X.java\n"
		"@@ -1,1 +1,1 @@\n"
		"-a\n+a \n"
		"@@ -10,0 +11,2 @@\n"
		"+x\n+y\n"
		"@@ -20,2 +22,0 @@\n"  # pure deletion -> skipped
	)
	assert fc.parse_hunks(diff) == [(1, 1), (11, 12)]


# --- exclude matching ---------------------------------------------------------

def test_is_excluded_by_full_path():
	p = "src/main/java/org/apache/sysds/conf/DMLConfig.java"
	assert fc.is_excluded(p, [p]) is True


def test_is_excluded_by_basename_glob():
	p = "src/main/java/org/apache/sysds/conf/DMLConfig.java"
	assert fc.is_excluded(p, ["*DMLConfig.java"]) is True


def test_is_excluded_no_match():
	assert fc.is_excluded("src/main/java/Foo.java", ["*DMLConfig.java"]) is False
	assert fc.is_excluded("src/main/java/Foo.java", []) is False


def test_load_excludes_skips_comments_and_blanks(tmp_path, monkeypatch):
	f = tmp_path / "exclude.txt"
	f.write_text("# a comment\n\n  \nsrc/main/java/Foo.java\n  *Bar.java  \n",
			encoding="utf-8")
	monkeypatch.setattr(fc, "EXCLUDE_FILE", str(f))
	assert fc.load_excludes() == ["src/main/java/Foo.java", "*Bar.java"]


def test_load_excludes_missing_file_returns_empty(tmp_path, monkeypatch):
	monkeypatch.setattr(fc, "EXCLUDE_FILE", str(tmp_path / "does-not-exist.txt"))
	assert fc.load_excludes() == []


# --- strip_src_prefix ---------------------------------------------------------

def test_strip_src_prefix():
	assert fc.strip_src_prefix("src/main/java/org/A.java") == "org/A.java"
	assert fc.strip_src_prefix("src/test/java/org/A.java") == "org/A.java"


# --- line_scoped_result: only edited-line formatting is kept ------------------

def test_line_scoped_result_keeps_only_edited_lines():
	# lines 1 and 3 are mis-formatted; an unchanged line 2 separates them so
	# difflib yields distinct hunks. Only line 3 is "edited", so line 1 (a
	# pre-existing violation the PR did not touch) must stay original.
	original = "int a=1;\nint ok = 0;\nint b=2;\n"
	formatted = "int a = 1;\nint ok = 0;\nint b = 2;\n"
	result, kept = fc.line_scoped_result(original, formatted, [(3, 3)])
	assert result == "int a=1;\nint ok = 0;\nint b = 2;\n"
	assert len(kept) == 1


def test_line_scoped_result_no_edited_lines_is_noop():
	original = "int a=1;\nint b=2;\n"
	formatted = "int a = 1;\nint b = 2;\n"
	result, kept = fc.line_scoped_result(original, formatted, [])
	assert result == original
	assert kept == []
