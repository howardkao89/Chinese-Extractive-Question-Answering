python QA_pred.py \
  --model_name_or_path ./model/QA \
  --test_file ./pred/MC/test_with_relevant.json \
  --context_file ./dataset/context.json \
  --max_seq_length 512 \
  --output_path ./pred/QA/prediction.csv
