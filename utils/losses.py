from asteroid.losses import singlesrc_neg_sisdr

def neg_sisdr_loss_wrapper(est_targets, targets):
    return singlesrc_neg_sisdr(est_targets[:,0], targets[:,0]).mean()