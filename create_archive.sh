#!/bin/bash

#
# file: create_archive.sh
# author: Samuel Kudla <xkudlas00@stud.fit.vutbr.cz>
# date: 12.05.2026
#

zip -r proj.zip . \
  -x "__pycache__/*" \
  "*.pyc" \
  ".vscode/*" \
  ".git/*" \
  ".DS_Store"

echo "Archive created: proj.zip"