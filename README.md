# CSIE5431 Applied Deep Learning Homework 1
* Name: 高榮浩
* ID: R12922127

## Environment
* Ubuntu 20.04
* GeForce RTX™ 2080 Ti 11G
* Python 3.9
* CUDA 11.8

## Download
```sh
bash ./download.sh
```

The context for ```download.sh``` is as follows.

```sh
gdown --folder 1qfyZrlOmg-BEXL08h830YuaxELVtw7tC

gdown --folder 1zqRG8V2U1oTDtzxNiQFkc26iVartMgL_
```

## Training
### Arguments
* ```model_name_or_path```: Path to pretrained model.
* ```train_file```: A json file containing the training data.
* ```validation_file```: A json file containing the validation data.
* ```context_file```: A json file containing the context data.
* ```max_seq_length```: The maximum total input sequence length after tokenization.
* ```per_device_train_batch_size```: Batch size (per device) for the training dataloader.
* ```gradient_accumulation_steps```: Number of updates steps to accumulate before performing a backward/update pass.
* ```learning_rate```: Initial learning rate to use.
* ```num_train_epochs```: Total number of training epochs to perform.
* ```output_dir```: Where to store the final model.

### Paragraph Selection
```sh
bash ./MC_train.sh
```

The context for ```MC_train.sh``` is as follows.

```sh
python MC_train.py \
  --model_name_or_path {...} \
  --train_file {...} \
  --validation_file {...} \
  --context_file {...} \
  --max_seq_length {...} \
  --per_device_train_batch_size {...} \
  --gradient_accumulation_steps {...} \
  --learning_rate {...} \
  --num_train_epochs {...} \
  --output_dir {...}
```

The arguments I used are as shown in the following table.

| Argument | Value |
|:--------:|:-----:|
| model_name_or_path | hfl/chinese-roberta-wwm-ext |
| train_file | ./dataset/train.json |
| validation_file | ./dataset/valid.json |
| context_file | ./dataset/context.json |
| max_seq_length | 512 |
| per_device_train_batch_size | 2 |
| gradient_accumulation_steps | 8 |
| learning_rate | 1e-5 |
| num_train_epochs | 3 |
| output_dir | ./model/MC |

### Span Selection (Extractive QA)
```sh
bash ./QA_train.sh
```

The context for ```QA_train.sh``` is as follows.

```sh
python QA_train.py \
  --model_name_or_path {...} \
  --train_file {...} \
  --validation_file {...} \
  --context_file {...} \
  --max_seq_length {...} \
  --per_device_train_batch_size {...} \
  --gradient_accumulation_steps {...} \
  --learning_rate {...} \
  --num_train_epochs {...} \
  --output_dir {...}
```

The arguments I used are as shown in the following table.

| Argument | Value |
|:--------:|:-----:|
| model_name_or_path | hfl/chinese-roberta-wwm-ext-large |
| train_file | ./dataset/train.json |
| validation_file | ./dataset/valid.json |
| context_file | ./dataset/context.json |
| max_seq_length | 512 |
| per_device_train_batch_size | 2 |
| gradient_accumulation_steps | 8 |
| learning_rate | 1e-5 |
| num_train_epochs | 9 |
| output_dir | ./model/QA |

## Prediction
### Arguments
* ```model_name_or_path```: Path to pretrained model.
* ```test_file```: A json file containing the testing data.
* ```context_file```: A json file containing the context data.
* ```max_seq_length```: The maximum total input sequence length after tokenization.
* ```output_path```: Where to store the final model.

### Paragraph Selection
```sh
bash ./MC_pred.sh
```

The context for ```MC_pred.sh``` is as follows.

```sh
python MC_pred.py \
  --model_name_or_path {...} \
  --test_file {...} \
  --context_file {...} \
  --max_seq_length {...} \
  --output_path {...}
```

The arguments I used are as shown in the following table.

| Argument | Value |
|:--------:|:-----:|
| model_name_or_path | ./model/MC |
| test_file | ./dataset/test.json |
| context_file | ./dataset/context.json |
| max_seq_length | 512 |
| output_path | ./pred/MC/test_with_relevant.json |

### Span Selection (Extractive QA)
```sh
bash ./QA_pred.sh
```

The context for ```QA_pred.sh``` is as follows.

```sh
python QA_pred.py \
  --model_name_or_path {...} \
  --test_file {...} \
  --context_file {...} \
  --max_seq_length {...} \
  --output_path {...}
```

The arguments I used are as shown in the following table.

| Argument | Value |
|:--------:|:-----:|
| model_name_or_path | ./model/QA |
| test_file | ./dataset/test.json |
| context_file | ./dataset/context.json |
| max_seq_length | 512 |
| output_path | ./pred/QA/prediction.csv |

### Complete in one go
```sh
bash ./run.sh /path/to/context.json /path/to/test.json /path/to/pred/prediction.csv
```

The context for ```run.sh``` is as follows.

```sh
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

```
