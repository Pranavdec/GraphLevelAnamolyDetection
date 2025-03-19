import torch

def loss_func(adj, A_hat, attrs, X_hat):
    # Attribute reconstruction loss
    diff_attribute = torch.pow(X_hat - attrs, 2)
    attribute_reconstruction_errors = torch.sqrt(torch.sum(diff_attribute, 1)) + 1e-6
    attribute_cost = torch.mean(attribute_reconstruction_errors)

    # structure reconstruction loss
    diff_structure = torch.pow(A_hat - adj, 2)
    structure_reconstruction_errors = torch.sqrt(torch.sum(diff_structure, 1)) + 1e-6
    structure_cost = torch.mean(structure_reconstruction_errors)


    return structure_cost, attribute_cost


def loss_cal(x, x_aug):
    # Contrastive Loss
    T = 0.2
    batch_size, _ = x.size()
    x_abs = x.norm(dim=1).clamp(min=1e-6)
    x_aug_abs = x_aug.norm(dim=1).clamp(min=1e-6)
    sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
    sim_matrix = torch.exp(sim_matrix / T).clamp(min=1e-6)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    denominator = (sim_matrix.sum(dim=1) - pos_sim).clamp(min=1e-6)

    loss = pos_sim / denominator
    loss = -torch.log(loss.clamp(min=1e-6)).mean()
    return loss