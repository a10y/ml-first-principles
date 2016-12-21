-- Implementation
require('torch')
require('sgd')
require('scale')

function mseLoss(w, x, y)
  return x * (w*x - y)
end

-- Simple regression over 1-dimensional data
linearModel = SGD.new{size=1, grad=mseLoss, eta=0.01, eps=0.001}

numPoints = 10 -- number of points to generate for each model

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
local scaler = NormalScaler.new()
X, off, scale = scaler:scale(linearPoints, 1)
print(X)
print(off)
print(scale)
--scaler.scale(quadraticPoints, 1)

linearModel:train(linearPoints, linearLabels)
print(linearModel.W)

-- try predicting next thing
p = torch.Tensor{1,50}
-- should guess 48
p:csub(off)
p:cdiv(scale)
print(linearModel.W * p)

--quadraticModel = SGD.new{size=2, grad=mseLoss, eta=0.001, eps=0.0005}
--quadraticModel:train(quadraticPoints, quadraticLabels)
--print(quadraticModel.W)
