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
        train_correct = 0
        train_loss = []

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            batch_size, num_frames, C, H, W = data.shape

            # flattening batch and frames to shape [batch_size*num_frames, 3, H, W]
            data = data.view(batch_size * num_frames, C, H, W)

            optimizer.zero_grad()

            # forward pass through model (output will have shape [batch_size*num_features, num_classes])
            output = model(data)
     
            # reshaping back to [batch_size, num_frames, num_classes]
            output = output.view(batch_size, num_frames, -1)
     
            # computing the average score for each class across the frames for each video (shape [batch_size, num_classes])
            average_output = torch.mean(output, dim=1)
    
            # computing the loss per video
            loss = criterion(average_output, target)
            loss.backward()
            optimizer.step()

            # computing the train loss
            train_loss.append(loss.item())

            # computing predictions for each video in the batch
            predicted = average_output.argmax(dim=1)

            # counting the number of correct predictions
            train_correct += (target==predicted).sum().cpu().item()

        # computing the val accuracy
        val_losses = []
        val_correct = 0
        model.eval()
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                batch_size, num_frames, C, H, W = data.shape
                data = data.view(batch_size * num_frames, C, H, W)
                output = model(data)
                output = output.view(batch_size, num_frames, -1)
                average_output = torch.mean(output, dim=1)

                val_losses.append(criterion(average_output, target).cpu().item())
                predicted = average_output.argmax(dim=1)
                val_correct += (target==predicted).sum().cpu().item()

        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)

        out_dict['train_acc'].append(train_correct/(len(train_loader.dataset)))
        out_dict['val_acc'].append(val_correct/(len(val_loader.dataset)))
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

    # computing the test accuracy
    test_losses = []
    test_correct = 0
    model.eval()
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            batch_size, num_frames, C, H, W = data.shape
            data = data.view(batch_size * num_frames, C, H, W)
            output = model(data)
            output = output.view(batch_size, num_frames, -1)
            average_output = torch.mean(output, dim=1)

            test_losses.append(criterion(average_output, target).cpu().item())
            predicted = average_output.argmax(dim=1)
            test_correct += (target==predicted).sum().cpu().item()

    test_loss = np.mean(test_losses)
    test_acc = test_correct/(len(dataloader.dataset))
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

    criterion = nn.NLLLoss()
    out_dict = {'train_acc': [],
                'val_acc': [],
                'train_loss': [],
                'val_loss': [],
                'n_epochs': 0}
    prev_val_loss = 1e10
  
    for epoch in range(num_epochs):
        model.train()
        train_correct = 0
        train_loss = []

        for frame_data, flow_data, target in train_loader:
            frame_data, flow_data, target = frame_data.to(device), flow_data.to(device), target.to(device)
            # zero the gradients computed for each weight
            optimizer.zero_grad()

            # reshaping the flow data from a 5D-tensor to a 4D-tensor
            flow_data = flow_data.view(flow_data.size(0), -1, flow_data.size(3), flow_data.size(4))

            # passing the frame and the flow stack through the dual stream network
            output_frame = model(frame_data, conv_type="frame")
            output_flow = model(flow_data, conv_type="flow")

            # computing the average output 
            average_output = (output_frame + output_flow) / 2
            
            # computing the loss
            loss = criterion(average_output, target)

            # backward pass through the network
            loss.backward()

            # updating the weights
            optimizer.step()

            # computing train loss
            train_loss.append(loss.item())

            # computing predictions for each video in the batch
            _, predicted_indices = torch.max(average_output, dim=1)
            predicted = predicted_indices.view(1, -1)

            # counting the number of correct predictions
            train_correct += (target==predicted).sum().cpu().item()
            
        # computing the val accuracy
        model.eval()
        val_losses = []
        val_correct = 0
        with torch.no_grad():
            for frame_data, flow_data, target in val_loader:
                frame_data, flow_data, target = frame_data.to(device), flow_data.to(device), target.to(device)
                flow_data = flow_data.view(flow_data.size(0), -1, flow_data.size(3), flow_data.size(4))

                output_frame = model(frame_data, conv_type="frame")
                output_flow = model(flow_data, conv_type="flow")
                average_output = (output_frame + output_flow) / 2

                val_losses.append(criterion(average_output, target).cpu().item())
                _, predicted_indices = torch.max(average_output, dim=1)
                predicted = predicted_indices.view(1, -1)
                val_correct += (target==predicted).sum().cpu().item()
        
        # average val loss
        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)

        # saving evaluation metrics in dictionary
        out_dict['train_acc'].append(train_correct/(len(train_loader.dataset)))
        out_dict['val_acc'].append(val_correct/(len(val_loader.dataset)))
        out_dict['train_loss'].append(np.mean(train_loss))
        out_dict['val_loss'].append(val_loss)

        # printing the 
        print(f"Loss train: {np.mean(train_loss):.3f}\t val: {val_loss:.3f}\t",
              f"Accuracy train: {out_dict['train_acc'][-1]*100:.1f}%\t val: {out_dict['val_acc'][-1]*100:.1f}%")
        
        # incrementing the number of epochs 
        out_dict['n_epochs'] += 1

        # checking if early stopping should be applied
        if val_loss > prev_val_loss and epoch + 1 > patience:
            print(f"Current validation loss ({val_loss}) is larger than for the previous epoch ({prev_val_loss})!")
            print("Early stopping applies!")
            break
        
        # saving the current model
        else:
            print(f"Validation loss improved from {prev_val_loss:.4f} to {val_loss:.4f}. Saving model...")
            torch.save(model.state_dict(), 'best_model.pt')

        # setting the prev_val_loss to the current val_loss
        prev_val_loss = val_loss

    return out_dict


def eval_dual_stream(device, model, dataloader):
    criterion = nn.NLLLoss()
    test_losses = []
    test_correct = 0
    model.eval()

    for frame_data, flow_data, target in dataloader:
        frame_data, flow_data, target = frame_data.to(device), flow_data.to(device), target.to(device)
        
        # reshaping the flow data from a 5D-tensor to a 4D-tensor
        flow_data = flow_data.view(flow_data.size(0), -1, flow_data.size(3), flow_data.size(4))

        with torch.no_grad():
            # passing the frame and the flow stack through the dual stream network
            output_frame = model(frame_data, conv_type="frame")
            output_flow = model(flow_data, conv_type="flow")

            # computing the average output 
            average_output = (output_frame + output_flow) / 2
        
        # computing test loss
        test_losses.append(criterion(average_output, target).cpu().item())

        # computing predictions for each video in the batch
        _, predicted_indices = torch.max(average_output, dim=1)
        predicted = predicted_indices.view(1, -1)

        # counting the number of correct predictions
        test_correct += (target==predicted).sum().cpu().item()

    # computing loss and accuracy and saving to dictionary
    test_loss = np.mean(test_losses)
    test_acc = test_correct/(len(dataloader.dataset))
    eval_results = {"Test loss": test_loss,
                    "Test accuracy": test_acc}
    
    return eval_results