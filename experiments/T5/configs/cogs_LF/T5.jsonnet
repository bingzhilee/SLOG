local data_base_url = "../../data/cogs_LF/";
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
        "add_prefix": false,
    },
    "model": {
        "type": "modified_t5",
        "model_name": model_name,
        "val_epoch": true,
        "postprocessor": {
            "type": "cogs",
        },
        "beam_search": {
            "max_steps": 1000,
            "beam_size": 4,
        }
    },
    "data_loader": {
        "batches_per_epoch": 500,
        "batch_sampler": {
            "type": "max_tokens_sampler",
            "max_tokens": 2048,
            "padding_noise": 0.1,
            "sorting_keys": ["source_tokens"]
        },
    },

    "validation_data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "batch_size": 64,
            "padding_noise": 0.0,
            "sorting_keys": ["target_tokens"]
        },
    },

    "trainer": {
        "num_epochs": 100,
        "optimizer": {
            "type": "adam",
            "lr": 1.5e-5,
        },
        "validation_metric": "+epochs",
        "num_gradient_accumulation_steps": 1,
        "cuda_device": 0,
        "callbacks": [
        {
            "type": "should_validate_callback",
            "validation_start": 90,
        }],
        "run_confidence_checks": false,
    },
    "evaluation":{
        "type": "cust_evaluator",
        "cuda_device": 0,
    },
}