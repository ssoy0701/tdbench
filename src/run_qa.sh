#!/bin/bash


python -m qa_pairs.convert_sql --task dyknow -q now --df_idx 0
python -m qa_pairs.convert_sql --task dyknow -q now --df_idx 1
python -m qa_pairs.convert_sql --task dyknow -q now --df_idx 2

python -m qa_pairs.convert_sql --task dyknow -q basic --df_idx 0
python -m qa_pairs.convert_sql --task dyknow -q basic --df_idx 1
python -m qa_pairs.convert_sql --task dyknow -q basic --df_idx 2

echo "All jobs processed!"