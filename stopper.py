class Stopper:
    def __init__(self, epoch=0, min_epoch=100, best_loss=None):
        self.epoch = epoch
        self.best_loss = best_loss
        self.counter = 0
        self.min_epoch = min_epoch

    def __call__(self, val_loss):
        self.epoch += 1
        if self.epoch < self.min_epoch:
            return False
        patience = self.epoch//4
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.counter >= patience:
                return True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return False

def test():
    stop = Stopper(epoch=200)
    loss = 10
    for i in range(300):
        if stop(loss): break
        loss += 0.01
        if i == 50: assert False
    print('passed')

if __name__ == "__main__":
    test()

