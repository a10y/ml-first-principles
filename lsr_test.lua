local LeastSquaresRegression = require('lsr')

function dLoss(w, x, y)
  return x * (w*x - y)
end

-- Simple linear regression over 1 variable.
model = LeastSquaresRegression.new(1, dLoss)

numPoints = 10
points = {}
labels = {}
for i=1,numPoints do
  points[i] = torch.Tensor{i}
  labels[i] =  (2*i) - 2 -- try and learn the 2's
end

model:train(points, labels)

print('W')
print(model.W)
