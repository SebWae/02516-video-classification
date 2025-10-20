import numpy as np
import random
import torch
import torch.nn as nn


def train_single_frame(model, train_loader, val_loader, device, optimizer, scheduler, num_epochs=10, patience=5):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    out_dict = {'train_acc': [],
                'val_acc': [],
                'train_loss': [],
                'val_loss': [],
                'n_epochs': 0}
    prev_val_loss = 1e10
  
    for epoch in range(num_epochs):
        model.train()
        #For each epoch
        train_correct = 0
        train_loss = []
        for _, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            #Zero the gradients computed for each weight
            optimizer.zero_grad()
            #Computing the video target (the target should be the same for all frames)
            video_target = target[0].unsqueeze(0)
            #Forward pass your image through the network
            output = model(data)
            #Computing the average score for each class across the frames (which should be from the same video)
            average_output = torch.mean(output, dim=0, keepdim=True)
            #Compute the loss
            loss = criterion(average_output, video_target)
            #Backward pass through the network
            loss.backward()
            #Update the weights
            optimizer.step()
            #Compute train loss
            train_loss.append(loss.item())
            #Compute how many were correctly classified
            predicted = average_output.argmax(dim=1)
            train_correct += (video_target==predicted).sum().cpu().item()

        #Compute the val accuracy
        val_losses = []
        val_correct = 0
        model.eval()
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            video_target = target[0].unsqueeze(0)
            with torch.no_grad():
                output = model(data)
                average_output = torch.mean(output, dim=0, keepdim=True)
            val_losses.append(criterion(average_output, video_target).cpu().item())
            predicted = average_output.argmax(dim=1)
            val_correct += (video_target==predicted).sum().cpu().item()
        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)
        out_dict['train_acc'].append(train_correct/(len(train_loader)))
        out_dict['val_acc'].append(val_correct/(len(val_loader)))
        out_dict['train_loss'].append(np.mean(train_loss))
        out_dict['val_loss'].append(val_loss)
        print(f"Loss train: {np.mean(train_loss):.3f}\t val: {val_loss:.3f}\t",
              f"Accuracy train: {out_dict['train_acc'][-1]*100:.1f}%\t val: {out_dict['val_acc'][-1]*100:.1f}%")
        
        # increment the number of epochs 
        out_dict['n_epochs'] += 1

        # check if early stopping should be applied
        if val_loss > prev_val_loss and epoch + 1 > patience:
            print(f"Current validation loss ({val_loss}) is larger than for the previous epoch ({prev_val_loss})!")
            print("Early stopping applies!")
            break
        
        # save current model
        else:
            print(f"Validation loss improved from {prev_val_loss:.4f} to {val_loss:.4f}. Saving model...")
            torch.save(model.state_dict(), 'best_model.pt')

        # setting the prev_val_loss to the current val_loss
        prev_val_loss = val_loss

    return out_dict


def eval(device, model, dataloader):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    #Compute the test accuracy
    test_losses = []
    test_correct = 0
    model.eval()
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        video_target = target[0].unsqueeze(0)
        with torch.no_grad():
            output = model(data)
            average_output = torch.mean(output, dim=0, keepdim=True)
        test_losses.append(criterion(average_output, video_target).cpu().item())
        predicted = average_output.argmax(dim=1)
        test_correct += (video_target==predicted).sum().cpu().item()

    test_loss = np.mean(test_losses)
    test_acc = test_correct/(len(dataloader))
    eval_results = {"Test loss": test_loss,
                    "Test accuracy": test_acc}
    
    return eval_results


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make CUDA operations deterministic (may slow down training slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_dual_stream(model, train_loader, val_loader, device, optimizer, scheduler, num_epochs=10, patience=5):
    """
    Function to train the dual stream network introduced by K. Simonyang and A. Zisserman
    """

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    out_dict = {'train_acc': [],
                'val_acc': [],
                'train_loss': [],
                'val_loss': [],
                'n_epochs': 0}
    prev_val_loss = 1e10
  
    for epoch in range(num_epochs):
        model.train()
        #For each epoch
        train_correct = 0
        train_loss = []
        for _, (frame_data, flow_data, target) in enumerate(train_loader):
            frame_data, flow_data, target = frame_data.to(device), flow_data.to(device), target.to(device)
            #Zero the gradients computed for each weight
            optimizer.zero_grad()

            #Computing the video target (the target should be the same for all frames)
            video_target = target[0].unsqueeze(0)

            # reshaping the flow data from a 5D-tensor to a 4D-tensor
            flow_data = flow_data.view(flow_data.size(0), -1, flow_data.size(3), flow_data.size(4))

            #Forward pass your image through the network
            output_frame = model(frame_data, conv_type="frame")
            output_flow = model(flow_data, conv_type="flow")

            #Computing the average score for each class across the frames (which should be from the same video)
            average_output = (output_frame + output_flow) / 2
            average_output_video = torch.mean(average_output, dim=0, keepdim=True)

            #Compute the loss
            loss = criterion(average_output_video, video_target)

            #Backward pass through the network
            loss.backward()

            #Update the weights
            optimizer.step()

            #Compute train loss
            train_loss.append(loss.item())

            #Compute how many were correctly classified
            predicted = average_output_video.argmax(dim=1)
            train_correct += (video_target==predicted).sum().cpu().item()

        #Compute the val accuracy
        model.eval()
        val_losses = []
        val_correct = 0

        for frame_data, flow_data, target in val_loader:
            frame_data, flow_data, target = frame_data.to(device), flow_data.to(device), target.to(device)
            video_target = target[0].unsqueeze(0)
            flow_data = flow_data.view(flow_data.size(0), -1, flow_data.size(3), flow_data.size(4))
            with torch.no_grad():
                output_frame = model(frame_data, conv_type="frame")
                output_flow = model(flow_data, conv_type="flow")
                average_output = (output_frame + output_flow) / 2
                average_output_video = torch.mean(average_output, dim=0, keepdim=True)
            val_losses.append(criterion(average_output_video, video_target).cpu().item())
            predicted = average_output_video.argmax(dim=1)
            val_correct += (video_target==predicted).sum().cpu().item()
        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)
        out_dict['train_acc'].append(train_correct/(len(train_loader)))
        out_dict['val_acc'].append(val_correct/(len(val_loader)))
        out_dict['train_loss'].append(np.mean(train_loss))
        out_dict['val_loss'].append(val_loss)
        print(f"Loss train: {np.mean(train_loss):.3f}\t val: {val_loss:.3f}\t",
              f"Accuracy train: {out_dict['train_acc'][-1]*100:.1f}%\t val: {out_dict['val_acc'][-1]*100:.1f}%")
        
        # increment the number of epochs 
        out_dict['n_epochs'] += 1

        # check if early stopping should be applied
        if val_loss > prev_val_loss and epoch + 1 > patience:
            print(f"Current validation loss ({val_loss}) is larger than for the previous epoch ({prev_val_loss})!")
            print("Early stopping applies!")
            break
        
        # save current model
        else:
            print(f"Validation loss improved from {prev_val_loss:.4f} to {val_loss:.4f}. Saving model...")
            torch.save(model.state_dict(), 'best_model.pt')

        # setting the prev_val_loss to the current val_loss
        prev_val_loss = val_loss

    return out_dict


def eval_dual_stream(device, model, dataloader):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    test_losses = []
    test_correct = 0
    model.eval()

    for frame_data, flow_data, target in dataloader:
        frame_data, flow_data, target = frame_data.to(device), flow_data.to(device), target.to(device)
        video_target = target[0].unsqueeze(0)
        flow_data = flow_data.view(flow_data.size(0), -1, flow_data.size(3), flow_data.size(4))
        with torch.no_grad():
            output_frame = model(frame_data, conv_type="frame")
            output_flow = model(flow_data, conv_type="flow")
            average_output = (output_frame + output_flow) / 2
            average_output_video = torch.mean(average_output, dim=0, keepdim=True)
        test_losses.append(criterion(average_output_video, video_target).cpu().item())
        predicted = average_output_video.argmax(dim=1)
        test_correct += (video_target==predicted).sum().cpu().item()

    test_loss = np.mean(test_losses)
    test_acc = test_correct/(len(dataloader))
    eval_results = {"Test loss": test_loss,
                    "Test accuracy": test_acc}
    
    return eval_results