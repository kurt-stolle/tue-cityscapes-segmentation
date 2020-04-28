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
