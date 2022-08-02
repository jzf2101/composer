import argparse
import os

import numpy as np
import torch
from torch.nn.functional import cross_entropy

from composer.trainer.trainer_hparams import TrainerHparams

os.environ['LOCAL_WORLD_SIZE'] = '1'

#training_runs = ['bert-base-128-jzf-seed-0-o0hc',
#'bert-base-128-jzf-seed-23-cpur',
#'bert-base-128-jzf-seed-42-s5xi']

key = '{}/checkpoints/ep0-ba{}-rank0'

#iterations = np.arange(3500,68796, 3500).tolist() + [68796]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('checkpoint', type=int)
    parser.add_argument("run", help="run name", type=str)
    args = parser.parse_args()
    hparams = TrainerHparams.create(f='load_bert.yaml')
    hparams.load_path = key.format(args.run, args.checkpoint)
    trainer = hparams.initialize_object()
    trainer.state.model.eval()
    for batch_idx, trainer.state.batch in enumerate(trainer._iter_dataloader()):
        trainer.state.batch = trainer._device.batch_to_device(trainer.state.batch)
        trainer.state.batch = trainer._train_data_spec.device_transforms(trainer.state.batch)
        labels = trainer.state.batch.pop('labels')
        with torch.no_grad():
            output = trainer.state.model.forward(trainer.state.batch)
        losses = cross_entropy(output['logits'].view(-1, trainer.state.model.config.vocab_size),
                               labels.view(-1),
                               reduction='none').view_as(labels).cpu().detach().numpy()
        np.save('{}_iteration{}_batch{}.pt'.format(run, checkpoint, batch_idx), losses)
    trainer.close()
