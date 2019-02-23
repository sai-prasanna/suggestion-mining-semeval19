{
  "dataset_reader": {
    "type": "suggestion_mining",
    "oversample_n": 2
  },
  "train_data_path": "../../data/V1.4_Training.csv",
  "validation_data_path": "../../data/SubtaskA_Trial_Test_Labeled.csv",
  "test_data_path": "../../data/SubtaskA_EvaluationData_labeled.csv",
  "evaluate_on_test": true,
  "model": {
    "type": "text_classifier",
    "embedding_dropout": 0.3,
    "model_text_field_embedder": {
      "tokens": {
        "type": "embedding",
        "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.840B.300d.txt.gz",
        "embedding_dim": 300,
        "trainable": false
      }
    },
    "internal_text_encoder": {
      "type": "boe",
      "embedding_dim": 300,
      "averaged": true
    },
    "classifier_feedforward": {
      "input_dim": 300,
      "num_layers": 4,
      "hidden_dims": [300, 150, 75,  2],
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
