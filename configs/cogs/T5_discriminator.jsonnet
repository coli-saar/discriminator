local data_base_url = "data/discriminator/cogs/";
local train_data = data_base_url + "train.tsv";
local dev_data = data_base_url + "small_dev.tsv";
local test_data = data_base_url + "small_dev.tsv";
local model_name = "t5-base";

{
    "train_data_path": train_data,
    // "validation_data_path": dev_data,
    //"test_data_path": test_data,
    "dataset_reader": {
        "type": "cfq",
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
            "type": "simple",
        },
        "beam_search": {
            "max_steps": 5,
            "beam_size": 1,
        },
        "metrics": [
            {"type": "auc"},
            {"type": "acc"},
            {"type": "f1", positive_label: 0},
            {"type": "f1", positive_label: 1},
        ],
    },
    "data_loader": {
        "batches_per_epoch": 1000,
        "batch_sampler": {
            "type": "max_tokens_sampler",
            "max_tokens": 8096,
            "padding_noise": 0.1,
            "sorting_keys": ["source_tokens"]
        },
    },

    "validation_data_loader": {
        "type": "multiprocess",
        "batch_sampler":{
            "type": "bucket",
            "batch_size": 64,
            "padding_noise": 0.0,
            "sorting_keys": ["source_tokens"],
        }
    },

    "trainer": {
        "num_epochs": 100,
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 1e-5,
            "weight_decay": 0.001,
        },
        "validation_metric": "+acc",
        "patience": 10,
        "num_gradient_accumulation_steps": 2,
        "cuda_device": 0,
        "callbacks": [
        {
            "type": "debug_wandb",
            "summary_interval": 4000,
            "project": "seq2seq_compgen",
            "entity": "ykyao",
        },
        {
            "type": "should_validate_callback",
            "validation_start": 5,
        }
        ],
        "run_confidence_checks": false, // BART fails the NormalizationBiasVerification check
    },
    "evaluation":{
        "type": "cust_evaluator",
        "cuda_device": 0,
    },
}