using Plots
using StatsPlots
using CSV

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

plot_train_val(2,2975,"data/baseline/train_loss_lr0-01.csv","data/baseline/val_loss_lr0-01.csv")

plot_train_val(2,2975,"data/augment/train_loss_hflip.csv", "data/augment/val_loss_hflip.csv")

plot_train_val(2,2975,"data/threshold/train_loss.csv", "data/threshold/val_loss.csv")
plot_train_val(2,2975,"data/threshold/train_iou.csv", "data/threshold/val_iou.csv")

plot_train_val(2,2975,"data/reduced-dims/train_loss.csv", "data/reduced-dims/val_loss.csv")
plot_train_val(2,2975,"data/reduced-dims/train_iou.csv", "data/reduced-dims/val_iou.csv")

plot_train_val(2,2975,"data/dropout/train_loss.csv", "data/dropout/val_loss.csv")
plot_train_val(2,2975,"data/dropout/train_iou.csv", "data/dropout/val_iou.csv")

plot_train_val(2,2975,"data/iou-loss/train_loss.csv", "data/iou-loss/val_loss.csv")
plot_train_val(2,2975,"data/iou-loss/train_iou.csv", "data/iou-loss/val_iou.csv")

plot_train_val(2,2975,"data/alpha-edges/train_loss.csv", "data/alpha-edges/val_loss.csv")
plot_train_val(2,2975,"data/alpha-edges/train_iou.csv", "data/alpha-edges/val_iou.csv")

plot_train_val(2,2975,"data/bilinear-false/train_loss.csv", "data/bilinear-false/val_loss.csv")
plot_train_val(2,2975,"data/bilinear-false/train_iou.csv", "data/bilinear-false/val_iou.csv")

plot_train_val(2,2975,"data/optimized-lr/0-001/train_loss.csv", "data/optimized-lr/0-001/val_loss.csv")
plot_train_val(2,2975,"data/optimized-lr/0-001/train_iou.csv", "data/optimized-lr/0-001/val_iou.csv")
plot_train_val(2,2975,"data/optimized-lr/0-01/train_loss.csv", "data/optimized-lr/0-01/val_loss.csv")
plot_train_val(2,2975,"data/optimized-lr/0-01/train_iou.csv", "data/optimized-lr/0-01/val_iou.csv")
