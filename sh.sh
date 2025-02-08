#!/bin/bash
# commit_each.sh
# This script adds and commits each changed file individually.

# Ensure we're inside a git repository
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Not inside a git repository."
  exit 1
fi

# Process each file from git status (null-separated to handle spaces)
git status --porcelain -z | while IFS= read -r -d '' entry; do
  # The format is: XY<space><file>, so skip the first 3 characters to get the file name
  file="${entry:3}"
  echo "Processing: $file"
  git add "$file"
  git commit -m "Update $file"
done
