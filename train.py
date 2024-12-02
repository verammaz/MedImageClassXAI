import wandb
import numpy as np
from tqdm import tqdm 
import torch

class Trainer():

    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.model_path = ''



    def train_one_epoch(self, dataloader, t):
        self.model.train()

        batch_size = dataloader.batch_size
        size = len(dataloader.dataset)

        batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc='Train')

        num_correct = 0.0
        total_loss = 0.0

        # Iterate all the training data and pass them into the model
        for idx, (images, labels) in enumerate(dataloader):

            self.optimizer.zero_grad() # zero all the gradients of the variable

            images, labels = images.to(self.device), labels.to(self.device)

            # Forward propagation
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward propagation
            loss.backward()
            # Gradient descent
            self.optimizer.step()

            loss = float(loss.item())

            # Update no. of correct image predictions & loss as we iterate
            num_correct += int((torch.argmax(outputs, dim=1)==labels).sum())
            total_loss += loss

            # tqdm lets you add some details so you can monitor training as you train.
            batch_bar.set_postfix(
                acc = "{:.04f}%".format(num_correct/(batch_size*(idx+1))*100),
                loss = "{:.04f}".format(float(total_loss/(idx+1))),
            )

            batch_bar.update()

            wandb.log({"n_examples": (idx+1)*batch_size + size * t, "train_loss": loss, "train_accuracy": num_correct/(batch_size*(idx+1))*100})

        batch_bar.close()

        acc = num_correct/(batch_size*(idx+1))*100
        avg_loss = total_loss/(idx+1)

        return acc, avg_loss
    
    def train(self, n_epochs, train_dataloader, val_dataloader, model_path=None):
        print(f'Training on device: {self.device}')

        best_val_loss = np.inf
        best_val_acc = 0.0

        for t in range(n_epochs):
            print(f"\nEpoch {t+1}\n----------------------")
            train_acc, train_loss = self.train_one_epoch(train_dataloader, t)
            print(f'Train Loss: {train_loss}, Train Accuracy: {train_acc}')
            
            val_loss, val_acc, sens, spec = self.evaluate(val_dataloader, "Validation")
            wandb.log({"val_specificity": spec, "val_sensitivity": sens})
            print(f'Val Loss: {val_loss}, Val Accuracy: {val_acc}')
            
            #test_loss, test_acc, sens, spec = evaluate(model, test_dataloader, "Test", criterion)
            #wandb.log({"test_specificity": spec, "test_sensitivity": sens})
            #print(f'Test Loss: {test_loss}, Test Accuracy: {test_acc}')
            
            wandb.log({"epoch": t, "train_loss_": train_loss, 
                                    "val_loss_": val_loss, 
                                    "train_acc_": train_acc, 
                                    "val_acc_": val_acc, 
                                    #"test_loss_": test_loss, 
                                    #"test_acc_": test_acc
                                    })

            # Save the best model based on validation accuracy or loss
            if model_path is not None:
                self.model_path = model_path
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_acc = val_acc
                    torch.save(self.model.state_dict(), model_path)
                    print(f"Best model saved to {model_path}")


    def evaluate(self, dataloader, dataname, confusion_matrix=False):
        batch_size = dataloader.batch_size

        self.model.eval()
        batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, position=0, leave=False, desc=f'Evaluate on {dataname} dataset')

        total_loss, num_correct = 0, 0

        TP, TN, FP, FN = 0, 0, 0, 0

        with torch.no_grad():
            for idx, (images, labels) in enumerate(dataloader):

                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                preds = torch.argmax(outputs, dim=1)

                num_correct += int((preds==labels).sum())
                total_loss += float(loss.item())

                for p, l in zip(preds, labels):
                    if p == 1 and l == 1:
                        TP += 1
                    elif p == 0 and l == 0:
                        TN += 1
                    elif p == 1 and l == 0:
                        FP += 1
                    elif p == 0 and l == 1:
                        FN += 1

                batch_bar.set_postfix(
                    acc= "{:.04f}%".format(num_correct/(batch_size*(idx+1))*100),
                    loss= "{:.04f}".format(float(total_loss/(idx+1)))
                )

                batch_bar.update()

        batch_bar.close()

        acc = num_correct/(batch_size*(idx+1))*100
        avg_loss = total_loss/(idx+1)

        sensitivity, specificity =  TP / (TP + FN), TN / (TN + FP)

        if not confusion_matrix:
            return avg_loss, acc, sensitivity, specificity
        
        else:
            cm = np.array([[TP, FP], [FN, TN]])
            return avg_loss, acc, sensitivity, specificity, cm
    

    def final_evaluate(self, test_dataloader):
    
        # Reload the best model
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()

        print("Best Model Performance:\n-------------------")
        test_loss, test_acc, sensitivity, specificity, cm = self.evaluate(test_dataloader, "Test", confusion_matrix=True)
        print(f'Loss: {test_loss}, Accuracy: {test_acc}')
        print(f'Confusion Matrix\n: {cm}')
        print(f'Sensitivity: {sensitivity}, Specificity: {specificity}')
        wandb.log({"Sensitivity": sensitivity, "Specificity": specificity})