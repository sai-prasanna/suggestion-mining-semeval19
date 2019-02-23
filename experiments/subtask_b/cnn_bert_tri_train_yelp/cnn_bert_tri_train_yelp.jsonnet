{
  "dataset_reader": {
    "type": "suggestion_mining",
    "oversample_n": 2,
    "max_length": 300,
    "tokenizer": {
        "word_splitter": "bert-basic"
    },
    "token_indexers": {
      "bert": {
        "type": "bert-pretrained",
        "pretrained_model": "bert-base-uncased",
        "do_lowercase": true
      }
    }
  },
  "train_data_path": "",
  "validation_data_path": "../../data/SubtaskB_Trial_Test_Labeled.csv",
  "model": {
    "type": "text_classifier",
    "embedding_dropout": 0.5,
    "model_text_field_embedder": {
      "token_embedders": {
        "bert":{
            "type": "bert-pretrained",
            "pretrained_model": "bert-base-uncased",
            "top_layer_only": true,
        },
      },
      "embedder_to_indexer_map": {
            "bert": ["bert", "bert-offsets"]
      },
      "allow_unmatched_keys": true
    },
    "internal_text_encoder": {
      "type": "cnn",
      "embedding_dim": 768,
      "num_filters": 192,
    },
    "classifier_feedforward": {
      "input_dim": 768,
      "num_layers": 4,
      "hidden_dims": [768, 324, 162 , 2],
      "activations": ["relu", "relu", "relu", "linear"],
      "dropout":     [0.2, 0.2, 0.2, 0],
    }
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["text", "num_tokens"]],
    "batch_size": 128
  },
  "trainer": {
    "num_epochs": 50,
    "cuda_device": 0,
    "patience": 7,
    "learning_rate_scheduler": {
      "type": "reduce_on_plateau",
      "patience": 3
    },
    "validation_metric": "+f1",
    "optimizer": {
      "type": "adam"
    },
    "should_log_parameter_statistics": false,
    "should_log_learning_rate": true, 
    "num_serialized_models_to_keep": 1
  }
}
