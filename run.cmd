python build_configs.py

python run_language_modeling.py ^
    --output_dir=output ^
    --model_type=reformer ^
    --config_name=protein_reformer ^
    --tokenizer_name=protein_reformer ^
    --line_by_line ^
    --do_train ^
    --train_data_file=data\yeast\yeast_train.txt ^
    --num_train_epochs 100 ^
    --do_eval ^
    --eval_data_file=data\yeast\yeast_val.txt ^
    --evaluate_during_training ^
    --logging_steps 200 ^
    --block_size 4608 ^
    --overwrite_output_dir