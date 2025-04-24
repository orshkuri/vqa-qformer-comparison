import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(x)


x = torch.randn(5, 10)
w = nn.Parameter(torch.randn(10, 10), requires_grad=True)

model = Model()

# Freeze all model parameters
for param in model.parameters():
    param.requires_grad = False

optimizer = torch.optim.Adam([w], lr=0.01)
optimizer.zero_grad()
print("Initial weights:", w.grad)
z = x @ w
z = model(z)

loss = z.sum()
loss.backward()
optimizer.step()
print("Updated weights:", w.grad)

z = x @ w
z = model(z)

loss = z.sum()
loss.backward()
optimizer.step()
print('-------')
print("Updated weights:", w.grad)
