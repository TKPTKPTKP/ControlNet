from share import *

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from cityscape_dataset import Cityscape
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

# fix random seed
torch.manual_seed(42)

# Configs
resume_path = './models/control_sd21_ini.ckpt'
batch_size = 4
logger_freq = 1500
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v21.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = Cityscape()
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger], max_epochs=300)

# Deadlock when 1 epoch is over, need to set limit_train_batches to a int
# Set drop_last=True is useless according to Github issue https://github.com/Lightning-AI/lightning/issues/11910#issuecomment-1055121784
# Another solution is to set shuffle=False or set a fix seed for dataloader, to avoid unconsistency on different threads caused by randomness.
# See https://github.com/Lightning-AI/lightning/issues/10947#issuecomment-1058873056

# trainer = pl.Trainer(gpus=[0, 1, 2, 3], strategy="ddp", precision=32, callbacks=[logger], max_epochs=2)

# Train!
trainer.fit(model, dataloader)
