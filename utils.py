import os
import torch
import torch.nn as nn
from models.resnet_visdrone import ResNet44

#preparing logs folder and store folder : reason unknown 
def prepare_folders(args):
    folders = [args.root_log, args.root_model]
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
    store_dir = os.path.join(args.root_log, args.store_name)
    if not os.path.exists(store_dir):
        os.makedirs(store_dir)
    return store_dir

#adjusts the learning rate , for every 30 epoch it reduces the learning rate 
def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print(f"Epoch {epoch}, Learning Rate: {lr}")


def train(train_loader, model, optimizer, epoch, args, log_file, tf_writer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    print(f"\n🚂 Training loop started — Total Batches: {len(train_loader)}")

    for i, (images, targets) in enumerate(train_loader):
        print(f"\n📦 Batch {i}/{len(train_loader)} loaded")
        print(f"   → Image shape: {images.shape}, Target shape: {targets.shape}")
        
        images, targets = images.to(args.device), targets.to(args.device)
        print(f"   → Moved to {args.device}")

        optimizer.zero_grad()

        outputs = model(images)
        print(f"   → Forward pass done — Output shape: {outputs.shape}")

        loss = criterion(outputs, targets)
        print(f"   → Loss computed: {loss.item()}")  # per batch loss compute ho rha hai 

        loss.backward()
        optimizer.step()
        print(f"   → Backward + Optimizer step complete") #once per batch 

        # 👇 Add after the loss is computed
        print(f"   🔢 Batch Loss Value: {loss.item():.4f}") #loss per batch 

    total_loss += loss.item() #batch size average loss ko add kiya humne 

# 👇 Get predictions and show them
    _, predicted = outputs.max(1)
    print(f"   📌 Predicted labels: {predicted.tolist()}")
    print(f"   🎯 Actual targets  : {targets.tolist()}")

# 👇 Count batch size and correct predictions
    batch_total = targets.size(0)
    batch_correct = predicted.eq(targets).sum().item()

    total += batch_total
    correct += batch_correct

    print(f"   ✅ Correct in Batch: {batch_correct}/{batch_total}")
    print(f"   📈 Total so far    : {total}")
    print(f"   🎯 Correct so far  : {correct}")


    avg_loss = total_loss / len(train_loader) #loss of 1 epoch , after adding the loss of 324 batches 
    accuracy = 100. * correct / total
    print(f"\n✅ Epoch {epoch}: Avg Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%")

#writting in the writer 
    log_file.write(f"{epoch},{avg_loss},{accuracy}\n")
    tf_writer.add_scalar('train/loss', avg_loss, epoch)
    tf_writer.add_scalar('train/acc', accuracy, epoch)


def validate(val_loader, model, epoch, args, log_file, tf_writer):
    model.eval()
    correct = 0 #total correct predictions 
    total = 0 #total predictions seen so far 
    print("\n🧪 Validation loop started")
    
    with torch.no_grad(): #no gradient usage 
        for images, targets in val_loader:
            images, targets = images.to(args.device), targets.to(args.device)
            print(f"   → Validating batch — image shape: {images.shape}")
            
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item() #element wise check #sum the total true # item convert tensor in int in python 

    acc = 100. * correct / total
    print(f"✅ Validation Epoch {epoch} Accuracy: {acc:.2f}%")
    log_file.write(f"{epoch},{acc}\n")
    tf_writer.add_scalar('val/acc', acc, epoch)
    return acc


def save_checkpoint(args, state, is_best):
    checkpoint_dir = os.path.join(args.root_model, args.store_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    filename = os.path.join(checkpoint_dir, 'checkpoint.pth.tar')
    torch.save(state, filename)
    print(f"💾 Checkpoint saved to: {filename}")
    
    if is_best:
        best_filename = os.path.join(checkpoint_dir, 'model_best.pth.tar')
        torch.save(state, best_filename)
        print(f"🏆 Best model saved as: {best_filename}")


