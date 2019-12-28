from torchtools.loss import Loss
import pdb

loss = Loss([dict(line_detect=1.0)])
print(loss.losses)