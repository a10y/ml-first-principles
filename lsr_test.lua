-- Implementation
require('torch')
require('sgd')
require('scale')

function mseLoss(w, x, y)
  return x * (w*x - y)
end

-- Simple regression over 1-dimensional data
linearModel = SGD.new{size=2, grad=mseLoss, eta=0.01, eps=0.001, miniters=100}

numPoints = 1000 -- number of points to generate for each model

linearPoints = torch.Tensor(numPoints, 2)
linearLabels = torch.Tensor(numPoints)
quadraticPoints = torch.Tensor(numPoints, 3)
quadraticLabels = torch.Tensor(numPoints)

for x=1,numPoints do
  linearPoints[x] = torch.Tensor{1, x}
  linearLabels[x] = (2*x) - 2
  quadraticPoints[x] = torch.Tensor{1, x, x^2}
  quadraticLabels[x] = 2*x^2 - 2*x + 2
end

-- Standardize both matrices.
scaler = NormalScaler.new()
X, off, scale = scaler:scale(linearPoints, 1)

linearModel:train(linearPoints, linearLabels)

-- try predicting next thing
p = torch.Tensor{1,500}
-- should guess 48
p:csub(off)
p:cdiv(scale)
print(linearModel.W * p)

X, off, scale = scaler:scale(quadraticPoints, 1)
quadraticModel = SGD.new{size=3, grad=mseLoss, eta=0.01, eps=0.1, miniters=10}
quadraticModel:train(quadraticPoints, quadraticLabels)
p = torch.Tensor{1,500,250000}
p:csub(off)
p:cdiv(scale)
print(quadraticModel.W * p) -- should guesss 2*(500^2) - 2*(500) + 2 => 499,002
