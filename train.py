import torch
from tqdm import tqdm
from utils.loss import MS_SSIMLoss
from utils import isolate_cloud_formations


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train(ca, dataset, config):

    def train_step(x, target, steps, optimizer, scheduler):
        loss = 0
        gamma = 1.0
        for i in range(target.shape[1]):
            x = ca(x, steps=steps)
            loss += gamma**i * F.mse_loss(x[:, -1, ...], target[:, i, ...])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        return x, loss

    ca.train()
    loss_log = []

    for i in tqdm(range(10000+1)):
        for batch_coordinates, batch_features, batch_targets in dataset:
            x0 = isolate_cloud_formations(batch_features).to(device)
            target = isolate_cloud_formations(batch_targets).to(device)
            x, loss = train_step(ca, x0, target[:, :, ...], 10, config.optimizer, config.scheduler)
            # x, loss = train(x0[:, :1, :, :].to(device), target[:, :1, :, :].to(device), 300, optimizer, scheduler)

            step_i = len(loss_log)
            loss_log.append(loss.item())

            if step_i%1000 == 0:
                # clear_output()
                print(step_i, "loss =", loss.item())
                print(x0.shape, x0.mean().item(), x.mean().item())
                # visualize_batch(target[:, 0, ...].detach().cpu().numpy(), x[:, -1, ...].detach().cpu().numpy())
                # plot_loss(loss_log)
                torch.save(ca.state_dict(), config.model_path)
