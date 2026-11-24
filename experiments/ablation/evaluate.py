"""Linear probing on frozen DGMAE embeddings."""

import torch
from torch.nn.functional import cross_entropy

from graphmae.evaluation import LogisticRegression1
from graphmae.utils import accuracy, create_optimizer


def probe(embedding, labels, masks, config):
    train, validation, test = masks
    classifier = LogisticRegression1(embedding.shape[1], config.num_classes).to(embedding.device)
    optimizer = create_optimizer("adam", classifier, config.lr_f, config.weight_decay_f)
    train_x, train_y = embedding[train], labels[train]
    validation_x, validation_y = embedding[validation], labels[validation]
    test_x, test_y = embedding[test], labels[test]
    best_val = -1.0
    for epoch in range(config.max_epoch_f):
        logits = classifier(None, train_x)
        loss = cross_entropy(logits, train_y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            validation_acc = accuracy(classifier(None, validation_x), validation_y)
            if validation_acc >= best_val:
                best_val = validation_acc
                best_epoch = epoch + 1
                best_state = {key: value.detach().clone() for key, value in classifier.state_dict().items()}

    with torch.no_grad():
        final_test_acc = accuracy(classifier(None, test_x), test_y)
        classifier.load_state_dict(best_state)
        test_acc = accuracy(classifier(None, test_x), test_y)
    return {
        "validation_accuracy": best_val,
        "probe_epoch": best_epoch,
        "test_accuracy": test_acc,
        "final_test_accuracy": final_test_acc,
        "split_sizes": [int(mask.sum()) for mask in masks],
    }
