-- Simple linear regression using torch

require('torch')

-- class for linear regression
local P = torch.class('LinearRegression')

-- size: number of entries in the feature vectors
-- dLoss: gradient of the loss function w.r.t. self.W.
--        Takes w, x and y
function P:__init(size, dLoss)
  self.W = torch.randn(size+1) -- extra entry for the bias term
  self.eta = 0.001
  self.dLoss = dLoss
end

function P:train(points, labels)
  local eps = 0.000001
  local count = 0
  local delta = 0.0
  while true do
    local i = torch.random(#points)
    local x, y = torch.cat(torch.Tensor{1}, points[i]), labels[i]
    grad = self.dLoss(self.W, x, y)
    self.W = self.W - self.eta * grad
    delta = torch.norm(grad)
    print('Trying i='..i..' delta='..delta)
    if delta < eps then
      count = count + 1
      if count == 10 then break end -- termination condition
    else
      count = 0
    end
  end
end

function P:predict(x)
  return self.W * data
end

return P
