from core.Value import Value
from NN.NN import MLP

#input dataset
xs = [[Value(0.0), Value(0.0)], [Value(0.0), Value(1.0)], [Value(1.0), Value(0.0)], [Value(1.0), Value(1.0)]]

#output dataset
ys = [Value(0.0), Value(1.0), Value(1.0), Value(0.0)]

model = MLP(2,[4,4,1])

for epoch in range(1000):

    y_pred = [model(x) for x in xs]

    loss = sum((yout - ygt) ** 2  for yout, ygt in zip(y_pred,ys))

    for p in model.parameters():
        p.grad = 0.0

    loss.backward()

    for p in model.parameters():
        p.data -= 0.01 * p.grad
    
    print(f"Epoch: {epoch} and Loss: {loss}")

y_pred = [model(x) for x in xs]
print(f"The final results are as follows : {y_pred}")