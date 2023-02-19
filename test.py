from torch.utils.tensorboard import SummaryWriter
import time

flag = 20
name = 'runs/' + time.strftime('%Y-%m-%d-%H-%M_', time.localtime()) + str(flag)
writer = SummaryWriter(log_dir=name)

for i in range(1, 100):
    writer.add_scalar('test', i*2, i)
