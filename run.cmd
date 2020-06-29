cd data
python .\split_files.py
cd ..

python run_language_modeling.py ^
    --output_dir=output ^
    --model_type=reformer ^
    --tokenizer_name=google/reformer-crime-and-punishment ^
    --line_by_line ^
    --do_train ^
    --train_data_file=./data/wikitext2/wiki.train.raw ^
    --do_eval ^
    --eval_data_file=./data/wikitext2/wiki.test.raw
