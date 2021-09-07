import torch

from .legnn_model import L_GCL, LEGNN, unsorted_segment_mean
from data_loader import *

device = "cuda" if torch.cuda.is_available() else "cpu"


def train(train_loader, val_loader, model, optimizer, scheduler, loss_fn, train_config):
    for epoch in range(train_config["n_epochs"]):
        print(f"Epoch: {epoch}")

        total_loss = 0
        val_loss = 0
        tp = 0

        for i, batch in enumerate(train_loader):
            #print(f"Batch: {i}")
            optimizer.zero_grad()

            p, y = torch.squeeze(batch["p"].to(device)), batch["y"].to(device)

            # print(p.size())
            # non_empty_mask = p.abs().sum(dim = 0).bool()
            # p = p[:, non_empty_mask]

            n_nodes = p.size()[0]

            edges = get_edges(n_nodes)
            row, column = edges

            h, _ = L_GCL.compute_radials(edges, p)  # torch.zeros(n_nodes, 1)

            output, x = model(h, p, edges)

            # output, _ = L_GCL.compute_radials(edges, x)
            # output = torch.sigmoid(torch.mean(output).unsqueeze(0))

            output = torch.mean(output)
            output = torch.sigmoid(output)
            output = output.unsqueeze(0)

            # print(str(output) + '\t\t\t' + str(y))

            loss = loss_fn(output, y.float())
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        scheduler.step()
        print(f"Train Loss: {total_loss}")

        for i, batch in enumerate(val_loader):
            p, y = torch.squeeze(batch["p"].to(device)), batch["y"].to(device)
            n_nodes = p.size()[0]

            edges = get_edges(n_nodes)
            row, column = edges

            h, _ = L_GCL.compute_radials(edges, p)  # torch.zeros(n_nodes, 1)

            output, x = model(h, p, edges)

            # output, _ = L_GCL.compute_radials(edges, x)
            # output = torch.sigmoid(torch.mean(output).unsqueeze(0))

            output = torch.mean(output)
            output = torch.sigmoid(output)
            output = output.unsqueeze(0)

            prediction = output.round()

            loss = loss_fn(output, y.float())
            val_loss += loss.item()

            tp += (prediction == y).item()

        print(f"Val Loss: {val_loss}, Accuracy: {tp / len(val_loader)}")
        torch.save({
            'epoch'               : epoch,
            'model_state_dict'    : model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss'                : loss
        }, 'legnn.pt')



def main():

    train_file = '../test.h5'#'../train.h5'
    with pd.HDFStore(train_file, mode = 'r') as store:
        train_df = store['table']

    val_file = '../val.h5'
    with pd.HDFStore(train_file, mode = 'r') as store:
        val_df = store['table']

    all_p, all_y = build_dataset(train_df, 1000)
    train_dataset = JetDataset(all_p, all_y)
    train_loader = DataLoader(train_dataset)#, batch_size = 100, shuffle = True)

    val_all_p, val_all_y = build_dataset(val_df, 1000)
    val_dataset = JetDataset(val_all_p, val_all_y)
    val_loader = DataLoader(val_dataset)

    model = LEGNN(input_feature_dim = 1, message_dim = 32, output_feature_dim = 2, edge_feature_dim = 0, device = device, n_layers = 6)
    model.share_memory()

    # Train the network
    train_config = {"n_epochs": 200,
                   "lr": 1e-3,
                   "factor": 0.3,
                   "patience": 50}

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=train_config["factor"],
                                                           step_size=train_config["patience"])
    loss_fn = torch.nn.BCELoss()
    
    train(train_loader, val_loader, model, optimizer, scheduler, loss_fn, train_config)

if __name__ == '__main__':
    #torch.set_num_threads(2)
    main()
