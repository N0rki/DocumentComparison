import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
import os

with open("group_embeddings/centrality_by_year.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

group_year_series = defaultdict(dict)

for year, groups in raw_data.items():
    for group, values in groups.items():
        avg = np.mean(values)
        group_year_series[group][int(year)] = avg

group_time_series = {}
for group, year_data in group_year_series.items():
    sorted_years = sorted(year_data.items())
    years = [y for y, _ in sorted_years]
    values = [v for _, v in sorted_years]
    group_time_series[group] = (years, values)


class GRUForecast(nn.Module):
    def __init__(self, input_size=1, hidden_size=16, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out


window_size = 3
forecast_horizon = 3
results = {}

output_dir = "sdaf_forecasts"
os.makedirs(output_dir, exist_ok=True)

for group, (years, values) in group_time_series.items():
    if len(values) < window_size + forecast_horizon:
        continue

    scaler = MinMaxScaler()
    norm_values = scaler.fit_transform(np.array(values).reshape(-1, 1)).flatten()

    X, y = [], []
    for i in range(len(norm_values) - window_size - forecast_horizon + 1):
        X.append(norm_values[i:i + window_size])
        y.append(norm_values[i + window_size:i + window_size + forecast_horizon])

    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor(y, dtype=torch.float32)

    model = GRUForecast()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(300):
        model.train()
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y[:, 0:1])
        loss.backward()
        optimizer.step()

    model.eval()
    last_seq = torch.tensor(norm_values[-window_size:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    forecast = []
    current_input = last_seq

    for _ in range(forecast_horizon):
        with torch.no_grad():
            next_val = model(current_input)
        forecast.append(next_val.item())
        next_val = next_val.view(1, 1, 1)  # reshape to [batch, seq, feature]
        current_input = torch.cat((current_input[:, 1:, :], next_val), dim=1)

    forecast_values = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()
    future_years = list(range(years[-1] + 1, years[-1] + 1 + forecast_horizon))

    results[group] = dict(zip(future_years, forecast_values.tolist()))

    plt.figure(figsize=(8, 4))
    plt.plot(years, values, marker='o', label='Historical')
    plt.plot(future_years, forecast_values, marker='x', linestyle='--', label='Forecast')
    plt.title(f"SDAF: {group}")
    plt.xlabel("Year")
    plt.ylabel("Centrality")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{group}_forecast.png")
    plt.close()

with open(f"{output_dir}/multi_year_forecasts.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"✅ Forecasts completed. Check the '{output_dir}' folder for plots and results.")