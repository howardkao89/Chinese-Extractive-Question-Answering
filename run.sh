python MC_pred.py \
  --model_name_or_path ./MC_model \
  --test_file "${2}" \
  --context_file "${1}" \
  --max_seq_length 512 \
  --output_path ./pred/MC/test_with_relevant.json

python QA_pred.py \
  --model_name_or_path ./QA_model \
  --test_file ./pred/MC/test_with_relevant.json \
  --context_file "${1}" \
  --max_seq_length 512 \
  --output_path "${3}"
