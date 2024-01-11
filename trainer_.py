import torch
import wandb
from sklearn.metrics import average_precision_score
import numpy as np
from utils_ import *
from losses_ import *
from tqdm import tqdm

## Train
def val(model, criterion, val_loader, args, fabric=None):
    
    model.eval()
    val_loss = 0
    val_loop = tqdm(val_loader, leave=True)
    predict_np = []
    truth_np = []
    with torch.no_grad():
        for imgs, label in val_loop:
            imgs, label = imgs.float().to(args.device), label.float().to(args.device)
            # Forward & Loss
            predicted_label = model(imgs)
            loss = criterion(predicted_label.squeeze(1), label)
            
            APs = []
            label_np = label.cpu().detach().numpy()
            pred_np = nn.Sigmoid()(predicted_label.squeeze(1)).cpu().detach().numpy()

            predict_np += pred_np.tolist()
            truth_np += label_np.tolist()
            
            val_loss += loss.item()
            val_loop.set_description(f"Validation")
            val_loop.set_postfix(loss=loss.item())
            
        total_val_map = average_precision_score(np.array(truth_np), np.array(predict_np))
        
    total_val_loss = val_loss / len(val_loader)
    if fabric.global_rank == 0:
        print(f"val loss: {total_val_loss:.4f} val map = {total_val_map:.4f}\n")
    
    return total_val_loss, total_val_map



def train(model, criterion, optimizer, train_loader, args, now_time, scheduler=None, val_loader=None, Wandb=True, fabric=None):
    if Wandb and fabric.global_rank == 0:
        wandb.init(entity=args.entity, project=args.project,
                   name=args.model_name+'-'+args.loss_name+'-'+now_time,
                   notes=str(torch.cuda.get_device_name())+' x '+str(1),
                   config={
                    "seed":args.seed,
                    "img_size":args.img_size,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "learning_rate": args.lr,
                    "min_lr":args.min_lr,
                    "weight_decay":args.weight_decay,
                    "model_name":args.model_name,
                    "loss_name":args.loss_name,
                   })
    # 학습
    # prior = ComputePrior(train_loader.dataset.__getitem__(0)[1])
    for epoch in range(args.epochs):
        model.train()
        total_loss, map = 0, 0
        loop = tqdm(train_loader, leave=True)
        for iteration, batch in enumerate(loop):
            # Accumulate gradient -> 더 큰 batch_size로 학습 가능
            is_accumulating = ((iteration % args.grad_accumulation) != 0)

            # imgs, label = imgs.float().to(args.device), label.float().to(args.device)
            imgs, label = batch
            
            with fabric.no_backward_sync(model, enabled=is_accumulating):
                # Forward & Loss
                predicted_label = model(imgs)

                loss = criterion(predicted_label.squeeze(1), label)

                # Backpropagation
                fabric.backward(loss)
            
            # prior.update(predicted_label)

                APs = []
                label_np = label.cpu().detach().numpy()
                pred_np = nn.Sigmoid()(predicted_label.squeeze(1)).cpu().detach().numpy()

                for i in range(predicted_label.shape[1]):
                    APs.append(average_precision_score(label_np[:, i], pred_np[:, i]))
                ap = np.mean(APs)

            if not is_accumulating:
                optimizer.step()
                optimizer.zero_grad()
                # fabric.print(f'Iteration: {iteration}') # for GradAccumulation check
            
            total_loss += loss.item()
            loop.set_description(f"Train")
            loop.set_postfix(loss=loss.item(), ap=ap.item())
            map += ap.item()

        if fabric.global_rank == 0:
            print(f"Epoch {epoch + 1} train loss: {total_loss / len(train_loader):.4f} train map = {map / len(train_loader):.4f}")


        if scheduler != None:
            scheduler.step()
            
        if val_loader != None:
            val_loss, val_map = val(model, criterion, val_loader, args, fabric)
        else:
            val_loss, val_map = 0, 0
        
        if Wandb and fabric.global_rank == 0:
            wandb.log({"Epoch": epoch + 1,
                   "lr": optimizer.param_groups[0]["lr"],
                   "train loss": total_loss / len(train_loader),
                   "train map": map / len(train_loader),
                   "val loss": val_loss,
                   "val map": val_map})   