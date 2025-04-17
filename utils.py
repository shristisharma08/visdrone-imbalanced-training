import os
import torch
from losses import IBLoss

def prepare_folders(args):
    folders = [args.root_log, args.root_model]
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
    store_dir = os.path.join(args.root_log, args.store_name)
    if not os.path.exists(store_dir):
        os.makedirs(store_dir)
    return store_dir

def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print(f"Epoch {epoch}, Learning Rate: {lr}")

def train(train_loader, model, optimizer, epoch, args, tf_writer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    criterion = IBLoss(alpha=10000.)
    print(f" Training loop started — Total Batches: {len(train_loader)}")

    for i, (images, targets) in enumerate(train_loader):
        print(f" Batch {i}/{len(train_loader)} loaded")
        print(f"   -> Image shape: {images.shape}, Target shape: {targets.shape}")
        
        images, targets = images.to(args.device), targets.to(args.device)
        print(f"   -> Moved to {args.device}")

        optimizer.zero_grad()
        outputs, features = model(images)
        print(f"   -> Forward pass done — Output shape: {outputs.shape}")

        loss = criterion(outputs, targets, features)
        print(f"   -> Loss computed: {loss.item():.4f}")

        loss.backward()
        optimizer.step()
        print(f"   -> Backward + Optimizer step complete")

        total_loss += loss.item() #total loss including the 1 in the batch 

        _, predicted = outputs.max(1) 
        print(f"   Raw outputs (logits):\n{outputs}")
        batch_total = targets.size(0) #total number of samples in this single batch  
        batch_correct = predicted.eq(targets).sum().item() #correct in this single batch 
        total += batch_total 
        correct += batch_correct 

        print(f"    Predicted labels: {predicted.tolist()}")
        print(f"    Actual targets  : {targets.tolist()}")
        print(f"    Correct in Batch: {batch_correct}/{batch_total}")
        print(f"    Running Total    : {total}") #total in batch till now 
        print(f"    Running Correct  : {correct}") # correct in batchs till now

    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    print(f"\n Epoch {epoch}: Avg Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%")

    tf_writer.add_scalar('train/loss', avg_loss, epoch)
    tf_writer.add_scalar('train/acc', accuracy, epoch)

def validate(val_loader, model, epoch, args, tf_writer):
    model.eval()
    correct = 0
    total = 0
    print("\n Validation loop started")
    
    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(args.device), targets.to(args.device)
            print(f"   -> Validating batch — image shape: {images.shape}")
            
            outputs, _ = model(images)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    print(f" Validation Epoch {epoch} Accuracy: {acc:.2f}%")
    tf_writer.add_scalar('val/acc', acc, epoch)
    return acc

def save_checkpoint(args, state, is_best):
    checkpoint_dir = os.path.join(args.root_model, args.store_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    filename = os.path.join(checkpoint_dir, 'checkpoint.pth.tar')
    torch.save(state, filename)
    print(f" Checkpoint saved to: {filename}")
    
    if is_best:
        best_filename = os.path.join(checkpoint_dir, 'model_best.pth.tar')
        torch.save(state, best_filename)
        print(f" Best model saved as: {best_filename}")
