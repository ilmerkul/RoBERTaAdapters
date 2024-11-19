import time
import torch
from torch.nn.functional import one_hot


class RoBERTaTrainer(object):
    def __init__(self, model, lr):
        self.model = model
        self.lr = lr

        self.opt = self.get_optimizer(model.parameters(), lr)
        self.loss = self.get_loss()

    def get_optimizer(self, parameters, lr):
        optimizer = torch.optim.Adam(params=parameters, lr=lr)
        return optimizer

    def get_loss(self):
        loss_function = torch.nn.CrossEntropyLoss()
        return loss_function

    def get_accuracy(self, y_pred, targets):
        predictions = torch.log_softmax(y_pred, dim=-1).argmax(dim=-1)
        accuracy = (predictions == targets).sum() / len(targets)
        return accuracy

    def train(self, train_loader, epochs):
        total_time = 0

        device = self.model.roberta.device
        count_tags = self.model.lin.lin1.weight.shape[0]
        for epoch in range(epochs):
            interval = len(train_loader) // 5

            total_train_loss = 0
            total_train_acc = 0

            start = time.time()

            self.model.train()
            for batch_idx, batch in enumerate(train_loader):
                self.opt.zero_grad()

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch["token_type_ids"].to(device)
                labels = batch["ner_tags"].to(device=device, dtype=torch.uint8)
                # length = batch["len"].to(device)
                # labels = labels[torch.arange(batch_size), length]

                # print(input_ids.shape)
                outputs = self.model(input_ids,
                                     attention_mask=attention_mask,
                                     token_type_ids=token_type_ids)
                # outputs = outputs[torch.arange(batch_size), length]
                # print(outputs.shape, labels.shape)
                logits = one_hot(labels.long(),
                                 num_classes=count_tags).to(dtype=torch.float)
                loss = self.loss(outputs, logits)
                acc = self.get_accuracy(outputs, labels)

                total_train_loss += loss.item()
                total_train_acc += acc.item()

                loss.backward()
                self.opt.step()

                if (batch_idx + 1) % interval == 0:
                    print("Batch: %s/%s | "
                          "Training loss: %.4f | "
                          "accuracy: %.4f" % (batch_idx + 1,
                                              len(train_loader),
                                              loss,
                                              acc))

            train_loss = total_train_loss / len(train_loader)
            train_acc = total_train_acc / len(train_loader)

            end = time.time()
            hours, remainder = divmod(end - start, 3600)
            minutes, seconds = divmod(remainder, 60)

            print(f"Epoch: {epoch + 1} "
                  f"train loss: {train_loss:.4f} "
                  f"train acc: {train_acc:.4f}")
            print("Epoch time elapsed: "
                  "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),
                                                  int(minutes),
                                                  seconds))
            print("")

            total_time += (end - start)

        # Get the average time per epoch
        average_time_per_epoch = total_time / epochs
        hours, remainder = divmod(average_time_per_epoch, 3600)
        minutes, seconds = divmod(remainder, 60)

        print("Average time per epoch: "
              "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),
                                              int(minutes),
                                              seconds))

    def evaluate(self, test_loader):
        interval = len(test_loader) // 5

        total_test_loss = 0
        total_test_acc = 0

        device = self.model.roberta.device
        count_tags = self.model.lin.lin1.weight.shape[0]
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch["token_type_ids"].to(device)
                labels = batch["ner_tags"].to(device=device, dtype=torch.uint8)

                outputs = self.model(input_ids,
                                     attention_mask=attention_mask,
                                     token_type_ids=token_type_ids)
                logits = one_hot(labels.long(),
                                 num_classes=count_tags).to(dtype=torch.float)
                loss = self.loss(outputs, logits)
                acc = self.get_accuracy(outputs, labels)

                total_test_loss += loss.item()
                total_test_acc += acc.item()

                if (batch_idx + 1) % interval == 0:
                    print("Batch: %s/%s | "
                          "Test loss: %.4f | "
                          "accuracy: %.4f" % (batch_idx + 1,
                                              len(test_loader),
                                              loss,
                                              acc))

        test_loss = total_test_loss / len(test_loader)
        test_acc = total_test_acc / len(test_loader)

        print(f"Test loss: {test_loss:.4f} acc: {test_acc:.4f}")
        print("")
