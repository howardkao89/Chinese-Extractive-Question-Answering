python MC_pred.py \
  --model_name_or_path ./model/MC \
  --test_file ./dataset/test.json \
  --context_file ./dataset/context.json \
  --max_seq_length 512 \
  --output_path ./pred/MC/test_with_relevant.json
