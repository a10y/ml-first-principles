require('torch')

local NormalScaler = torch.class('NormalScaler')

--[[ Scale the 2-tensor `X` to have mean 0 and variance 1.

ARGS:
  - `X`: the 2-tensor matrix that is to be scaled

RETURNS:
  - `X`: the scaled matrix
  - `X_off`: the offset to subtract from every newly encountered example
  - `X_scale`: the element-wise division factor for new examples

]]
function NormalScaler:scale(X, skip)
  local points, features = X:size()[1], X:size()[2]
  local X_off = torch.zeros(features)
  local X_scale = torch.ones(features)
  if skip == nil then
    -- No skip index, just calculate by all categories
    X_off:copy(X:mean(1))
    X_scale:copy(X:std(1))

    local means = torch.expand(X_off:reshape(1, features), points)
    local stddevs = torch.expand(X_scale:reshape(1, features), points)

    X:csub(means)
    X:cdiv(stddevs)

  elseif type(skip) == 'number' then

    local off_skip = X_off:narrow(1, skip+1, features-skip)
    local scale_skip = X_scale:narrow(1, skip+1, features-skip)
    local X_skip = X:narrow(2, skip+1, features-skip) -- work on the narrowed portion of X

    -- copy over the mean from the section we want
    off_skip:copy(X_skip:mean(1))
    scale_skip:copy(X_skip:std(1))

    local means = off_skip:reshape(1, features-skip):expandAs(X_skip)
    local stddevs = scale_skip:reshape(1, features-skip):expandAs(X_skip)

    X_skip:csub(means)
    X_skip:cdiv(stddevs)

  else
    error('invalid value for skip '..skip..', expected number')
  end
  return X, X_off, X_scale
end

return NormalScaler
