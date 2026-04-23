import wandb

wandb.init(project="Prueba conexión wandb")

for i in range(10):
    wandb.log({"mi_metrica": i})