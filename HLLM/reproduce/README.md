To reproduce the results on these datasets, run the script `{dataset-name}_HLLM_new.sh`, for example, `amzn_books_HLLM_new.sh`
```bash
sbatch {dataset-name}_HLLM_new.sh
```

Configs are included in scripts, under `--config_file` argument. Model config is consistent for all experiments, which can be found in `HLLM/code/HLLM/HLLM.yaml`. Dataset and training config files can be found in `HLLM/code/overall`, specifically `LLM_deepspeed_amzn.yaml` and `LLM_deepspeed_ml.yaml`.