from train import evaluate, split


def test_training_reduces_loss_and_fits_the_sample(trained):
    for task, (_, losses, model, tokenizer, texts, label_ids) in trained.items():
        assert losses[-1] < losses[0] / 2, (task, losses)
        # On the training data itself a fine-tuned model should be close to perfect.
        assert evaluate(model, tokenizer, texts, label_ids)["accuracy"] >= 0.9, task


def test_evaluate_metrics(trained):
    _, _, model, tokenizer, texts, label_ids = trained["department"]
    metrics = evaluate(model, tokenizer, texts[:4], label_ids[:4])
    assert metrics["n"] == 4 and 0 <= metrics["macro_f1"] <= 1


def test_split_is_deterministic(rows):
    a_train, a_test = split(rows, 0.25)
    _, b_test = split(rows, 0.25)
    assert a_test == b_test and len(a_test) == len(rows) // 4 and len(a_train) + len(a_test) == len(rows)
