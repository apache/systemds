#!/usr/bin/env python3
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
# Line-scoped Java formatting against dev/CodeStyle_eclipse.xml.
#
# The Eclipse formatter only works on whole files, and the existing tree is not
# fully formatter-clean, so formatting a whole edited file would flag lines the
# PR never touched. This script therefore formats each changed file but keeps
# only the formatting changes that fall on the lines the PR edited (like
# clang-format-diff): it diffs the original against the fully-formatted version
# and restricts the result to the changed line ranges.
#
#   --check (default): print the changed-line formatting fixes and exit 1 if any.
#   --fix            : apply only the changed-line formatting fixes in place.
#
# Usage: dev/format_changed.py [--check|--fix] [base-ref]
#
import difflib
import os
import re
import subprocess
import sys

FMT_VERSION = "2.24.1"
CONFIG = "dev/CodeStyle_eclipse.xml"
SRC_RE = re.compile(r"^src/(main|test)/java/.+\.java$")


def sh(*args, check=False):
	return subprocess.run(args, capture_output=True, text=True, check=check).stdout


def resolve_base(explicit):
	if explicit:
		return explicit
	for ref in ("upstream/main", "origin/main", "main"):
		if subprocess.run(["git", "rev-parse", "--verify", "--quiet", ref],
				capture_output=True).returncode == 0:
			return ref
	sys.exit("Could not determine a base ref; pass one explicitly.")


def changed_ranges(mergebase, path):
	# line ranges (1-based, inclusive) touched on the working-tree side, i.e.
	# committed-on-branch plus staged/unstaged edits, relative to the base.
	out = sh("git", "diff", "-U0", mergebase, "--", path)
	ranges = []
	for line in out.splitlines():
		m = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", line)
		if m:
			start = int(m.group(1))
			count = int(m.group(2)) if m.group(2) is not None else 1
			if count > 0:
				ranges.append((start, start + count - 1))
	return ranges


def overlaps(i1, i2, ranges):
	# original-side region [i1, i2) (0-based half-open) vs 1-based inclusive ranges
	lo, hi = i1 + 1, i2  # convert to 1-based inclusive; insert (i1==i2) -> lo>hi
	for (s, e) in ranges:
		if i1 == i2:  # pure insertion between original lines i1 and i1+1
			if s - 1 <= i1 <= e:
				return True
		elif not (hi < s or lo > e):
			return True
	return False


def main():
	args = [a for a in sys.argv[1:]]
	mode = "check"
	if "--fix" in args:
		mode = "fix"
		args.remove("--fix")
	if "--check" in args:
		args.remove("--check")
	base = resolve_base(args[0] if args else None)

	os.chdir(sh("git", "rev-parse", "--show-toplevel").strip())
	mergebase = sh("git", "merge-base", base, "HEAD").strip() or base

	files = [f for f in sh("git", "diff", "--name-only", "--diff-filter=ACMR",
			mergebase, "--").splitlines() if SRC_RE.match(f)]
	# include untracked new java files too (local pre-commit use)
	files += [f for f in sh("git", "ls-files", "--others", "--exclude-standard").splitlines()
			if SRC_RE.match(f) and f not in files]
	if not files:
		print(f"No changed Java source files (base: {base}).")
		return 0

	# snapshot originals so we can format in place then restore exactly, without
	# disturbing any unrelated working-tree state
	original = {f: open(f, encoding="utf-8").read() for f in files}

	# capture the PR-edited line ranges BEFORE we reformat the working tree,
	# otherwise git would report the whole reformatted file as changed
	ranges_by_file = {}
	for f in files:
		ranges = changed_ranges(mergebase, f)
		if not ranges:
			base_has = subprocess.run(["git", "cat-file", "-e", f"{mergebase}:{f}"],
					capture_output=True).returncode == 0
			if not base_has:  # brand-new/untracked file: treat every line as edited
				ranges = [(1, max(1, len(original[f].splitlines())))]
		ranges_by_file[f] = ranges

	includes = ",".join(re.sub(r"^src/(main|test)/java/", "", f) for f in files)
	subprocess.run(["mvn", "-q", "-ntp", "-B",
			f"net.revelc.code.formatter:formatter-maven-plugin:{FMT_VERSION}:format",
			f"-Dconfigfile={os.getcwd()}/{CONFIG}",
			"-Dmaven.compiler.source=17", "-Dmaven.compiler.target=17",
			f"-Dformatter.includes={includes}"], check=True)

	formatted = {f: open(f, encoding="utf-8").read() for f in files}

	any_issue = False
	for f in files:
		a = original[f].splitlines(keepends=True)
		b = formatted[f].splitlines(keepends=True)
		ranges = ranges_by_file[f]

		sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
		result = []
		kept_hunks = []
		for tag, i1, i2, j1, j2 in sm.get_opcodes():
			if tag == "equal":
				result.extend(a[i1:i2])
			elif overlaps(i1, i2, ranges):
				result.extend(b[j1:j2])
				kept_hunks.append((tag, i1, i2, j1, j2, a[i1:i2], b[j1:j2]))
			else:
				result.extend(a[i1:i2])

		# In fix mode, always write the reconstructed content: for non-flagged
		# files this equals the original (undoing mvn's whole-file reformat);
		# for flagged files it applies only the changed-line formatting.
		if mode == "fix":
			with open(f, "w", encoding="utf-8") as out:
				out.write("".join(result))

		if not kept_hunks:
			continue
		any_issue = True
		if mode == "check":
			print(f"\n--- a/{f}")
			print(f"+++ b/{f}")
			for tag, i1, i2, j1, j2, olds, news in kept_hunks:
				print(f"@@ -{i1 + 1},{len(olds)} +{i1 + 1},{len(news)} @@ (changed lines)")
				for ln in olds:
					print("-" + ln.rstrip("\n"))
				for ln in news:
					print("+" + ln.rstrip("\n"))

	# in check mode we only inspected; restore every file to its original content
	if mode == "check":
		for f in files:
			with open(f, "w", encoding="utf-8") as out:
				out.write(original[f])

	if any_issue and mode == "check":
		print("\nERROR: the changes above are required on lines this PR edited "
			"(per dev/CodeStyle_eclipse.xml).")
		print("Fix locally with:  dev/format-changed.sh")
		return 1
	if any_issue and mode == "fix":
		print("Applied changed-line formatting. Review and commit the result.")
	else:
		print("All PR-edited Java lines are correctly formatted.")
	return 0


if __name__ == "__main__":
	sys.exit(main())
