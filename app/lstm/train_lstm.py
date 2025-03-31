import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from model_lstm import LinkPredictorLSTM
from generate_training_pairs import generate_training_pairs
import chromadb

def train_link_predictor():
    # Load data from ChromaDB
    client = chromadb.HttpClient(host="localhost", port=8000)
    collection = client.get_collection("research_documents")
    data = generate_training_pairs(collection)

    # Prepare dataset
    X = np.array([d["x"] for d in data])
    y = np.array([d["y"] for d in data])
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (batch, seq_len=1, input_dim)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Model setup
    model = LinkPredictorLSTM(embedding_dim=768)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    model.train()
    for epoch in range(10):
        total_loss = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "link_predictor_lstm.pt")
    print("✅ Model saved as link_predictor_lstm.pt")

if __name__ == "__main__":
    train_link_predictor()
