import torch
from utils import evaluate

def evaluate(model, dataloader):
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in dataloader:
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total


def retrain():
    print("🔁 Retraining started...")


    old_model = torch.load("model/model.pth")
    old_model.eval()

    # TODO: load your real datasets
    train_loader = ...
    val_loader = ...

    # Evaluate old model
    old_accuracy = evaluate(old_model, val_loader)

    # Train new model
    new_model = train_model(train_loader)

    # Evaluate new model
    new_accuracy = evaluate(new_model, val_loader)

    print(f"Old Acc: {old_accuracy}, New Acc: {new_accuracy}")

    # Replace only if better
    if new_accuracy > old_accuracy:
        torch.save(new_model, "model/model.pth")
        print("✅ Model updated!")
    else:
        print("❌ No improvement, keeping old model")


if __name__ == "__main__":
    retrain()