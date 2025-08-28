local data_base_url = "data/parser/cogs/";
local train_data = data_base_url + "train.tsv";
local dev_data = data_base_url + "dev.tsv";
local test_data = data_base_url + "gen.tsv";
local model_name = "t5-base";
local random_seed = 0;
{
    "random_seed": random_seed,
    "numpy_seed": random_seed,
    "pytorch_seed": random_seed,
    "train_data_path": train_data,
    "validation_data_path": dev_data,
    //"test_data_path": test_data,
    "dataset_reader": {
        "type": "cogs",
        "source_tokenizer": {
            "type": "pretrained_transformer",
            "model_name": model_name,
        },
        "source_token_indexers": {
            "tokens": {
                "type": "pretrained_transformer",
                "model_name": model_name,
                "namespace": "source_tokens"
            }
        },
        "target_tokenizer": {
            "type": "pretrained_transformer",
            "model_name": model_name,
        },
        "target_token_indexers": {
            "tokens": {
                "type": "pretrained_transformer",
                "model_name": model_name,
                "namespace": "target_tokens"
            }
        },
        // "source_max_tokens": 1022,
        // "target_max_tokens": 100,
        // "max_instances": 1000 // DEBUG setting
        add_prefix: false,
    },
    "model": {
        "type": "modified_t5",
        "model_name": model_name,
        "postprocessor": {
            "type": "cogs",
        },
        "beam_search": {
            "max_steps": 400,
            "beam_size": 4,
        },
        "metrics": [{"type": "epochs"}, {"type": "acc"}]
    },
    "data_loader": {
        "batches_per_epoch": 200,
        "batch_sampler": {
            "type": "max_tokens_sampler",
            "max_tokens": 2048,
            "padding_noise": 0.1,
            "sorting_keys": ["source_tokens"]
        },
    },

    "validation_data_loader": {
        "type": "simple",
        "batch_size": 128,
    },

    "trainer": {
        "num_epochs": 5,
        "optimizer": {
            "type": "adam",
            "lr": 1e-5,
        },
        "validation_metric": "+epochs",
        //"patience": 5,
        "num_gradient_accumulation_steps": 1,
        "cuda_device": 0,
        "checkpointer":{
            "keep_most_recent_by_count": 5,
        },
        "callbacks": [
        {
            "type": "debug_wandb",
            "summary_interval": 100,
            "project": "seq2seq_compgen",
            "entity": "ykyao",
        },
        {
            "type": "should_validate_callback",
            "validation_start": 0,
        }
        ],
        "run_confidence_checks": false, // BART fails the NormalizationBiasVerification check
    },
    "evaluation":{
        "type": "cust_evaluator",
        "cuda_device": 0,
    },
}