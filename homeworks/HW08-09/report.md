# HW08-09
## Dataset
- Name: CIFAR10
- Seed: 42
## Regularization experiments (E1–E4)
- E1: model=hidden=[512, 256, 128] act=relu dropout=0.0 bn=False, optimizer=Adam (lr=0.001), best_val_acc=0.5044, best_val_loss=1.4059
- E2: model=hidden=[512, 256, 128] act=relu dropout=0.3 bn=False, optimizer=Adam (lr=0.001), best_val_acc=0.4522, best_val_loss=1.5460
- E3: model=hidden=[512, 256, 128] act=relu dropout=0.0 bn=True, optimizer=Adam (lr=0.001), best_val_acc=0.5171, best_val_loss=1.3780
- E4: model=hidden=[512, 256, 128] act=relu dropout=0.0 bn=True, optimizer=Adam (lr=0.001), best_val_acc=0.5336, best_val_loss=1.3268

## LR and optimizers
- O1: optimizer=Adam (lr=0.1, momentum=0.0, weight_decay=0.0), best_val_acc=0.4694
- O2: optimizer=Adam (lr=1e-05, momentum=0.0, weight_decay=0.0), best_val_acc=0.4934
- O3: optimizer=SGD (lr=0.01, momentum=0.9, weight_decay=0.0001), best_val_acc=0.5083

## Figures
- `artifacts/figures/curves_best.png`
- `artifacts/figures/curves_lr_extremes.png`
