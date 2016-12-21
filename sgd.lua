--[[ Vanilla SGD implementation for regression.
]]
require('torch')


local SGD = torch.class('SGD')

--[[ Constructor for SGDRegression instances.

ARGS:
  - `args.size` : number of entries in each feature vector
  - `args.lr`  : (optional) learning rate
  - `args.thresh`: (optional) threshold for convergence

]]
function SGD:__init(args)
  self.W = torch.zeros(args.size+1) -- extra entry for the bias term
  self.lr = args.lr or 0.0001
  self.grad = args.grad
  self.thresh = args.thresh or 0.0001
  self.miniters = args.miniters or 10
end

--[[ Train the weights of the linear model using SGD.

ARGS:
  - `X`: the 2-tensor design matrix
  - `y`: The 1-tensor of labels

]]
function SGD:train(X, y)
  -- iterate through training examples in random order until convergence.
  local delta = 10 * self.thresh
  local count = 0 -- number of iterations below the threshold
  while true do
    local i = torch.random(X:size()[1])
    point, label = X[i], y[i]
    grad = self.grad(self.W, point, label)
    self.W = self.W - self.lr * grad
    delta = grad:norm()
    if delta < self.thresh then
      count = count + 1
      if count == self.miniters then
        break
      end
    else
      count = 0
    end
  end
end

return SGD
