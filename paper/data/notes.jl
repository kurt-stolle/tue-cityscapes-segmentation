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

    @df data_train_loss plot(:global_step / batches_per_epoch, :loss,
        label="Training")
    @df data_val_loss plot!(:global_step / batches_per_epoch, :loss,
        label="Validation")
    xlabel!("Epoch")
    ylabel!("Loss")
    xticks!(0:1:epochs)
end

plot_train_val(2,2975,"baseline/train_loss_lr0-01.csv","baseline/val_loss_lr0-01.csv")
