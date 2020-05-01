using Plots
using StatsPlots
using CSV

## Plot baseline results
epochs = 10

function plot_train_val(batch_size, samples,fname_train, fname_val)
    batch_size = 2
    samples = 2975
    batches_per_epoch = samples / batch_size

    data_train_loss = CSV.read(fname_train)
    data_val_loss = CSV.read(fname_val)

    @df data_train_loss plot(:global_step / batches_per_epoch, :value,
        label="Training")
    @df data_val_loss plot!(:global_step / batches_per_epoch, :value,
        label="Validation")

    xlabel!("Epoch")
    ylabel!("Value")
    
    xticks!(0:1:epochs)
end

plot_train_val(2,2975,"baseline/train_loss_lr0-01.csv","baseline/val_loss_lr0-01.csv")

plot_train_val(2,2975,"augment/train_loss_hflip.csv", "augment/val_loss_hflip.csv")

plot_train_val(2,2975,"threshold/train_loss.csv", "threshold/val_loss.csv")
plot_train_val(2,2975,"threshold/train_iou.csv", "threshold/val_iou.csv")

plot_train_val(2,2975,"reduced-dims/train_loss.csv", "reduced-dims/val_loss.csv")
plot_train_val(2,2975,"reduced-dims/train_iou.csv", "reduced-dims/val_iou.csv")

plot_train_val(2,2975,"dropout/train_loss.csv", "dropout/val_loss.csv")
plot_train_val(2,2975,"dropout/train_iou.csv", "dropout/val_iou.csv")

plot_train_val(2,2975,"iou-loss/train_loss.csv", "iou-loss/val_loss.csv")
plot_train_val(2,2975,"iou-loss/train_iou.csv", "iou-loss/val_iou.csv")

plot_train_val(2,2975,"alpha-edges/train_loss.csv", "alpha-edges/val_loss.csv")
plot_train_val(2,2975,"alpha-edges/train_iou.csv", "alpha-edges/val_iou.csv")

plot_train_val(2,2975,"bilinear-false/train_loss.csv", "bilinear-false/val_loss.csv")
plot_train_val(2,2975,"bilinear-false/train_iou.csv", "bilinear-false/val_iou.csv")

plot_train_val(2,2975,"optimized-lr/0-001/train_loss.csv", "optimized-lr/0-001/val_loss.csv")
plot_train_val(2,2975,"optimized-lr/0-001/train_iou.csv", "optimized-lr/0-001/val_iou.csv")
plot_train_val(2,2975,"optimized-lr/0-01/train_loss.csv", "optimized-lr/0-01/val_loss.csv")
plot_train_val(2,2975,"optimized-lr/0-01/train_iou.csv", "optimized-lr/0-01/val_iou.csv")
