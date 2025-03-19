from cluster import Cluster
import hydra

cluster_name = 'V100'
#cluster_name = 'HOI'

# exp = sys.argv[1]
# cluster_name = sys.argv[2]

cluster = Cluster(cluster_name)

# # only hybrid
# with hydra.initialize(config_path='./configs', version_base=None):
#     for w in [0.2, 0.3, 0.7, 0.8]:
#         args = ["exp_name=hybrid_%d" % (w*10), "model.w_con_loss="+str(w)]
#         cfg = hydra.compose(config_name="conf", overrides=args)
#         run = cluster.submit(args=args, title=cfg.exp_name)

# # only hybrid-contrastive
# with hydra.initialize(config_path='./configs', version_base=None):
#     for w in [0.2, 0.3, 0.7, 0.8]:
#         args = ["exp_name=contra_%d" % (w*10), "model.w_con_loss="+str(w), "model.label_bce=False"]
#         cfg = hydra.compose(config_name="conf", overrides=args)
#         run = cluster.submit(args=args, title=cfg.exp_name)

# only daug
with hydra.initialize(config_path='./configs', version_base=None):
    for w in [0.0]:
        args = ["exp_name=daugadd_%d" % (w*10), "model.w_con_loss="+str(w), "model.heatmap=add"]
        cfg = hydra.compose(config_name="conf", overrides=args)
        run = cluster.submit(args=args, title=cfg.exp_name)

# both
with hydra.initialize(config_path='./configs', version_base=None):
    for w in [0.7, 0.8]:
        args = ["exp_name=daugadd_hybrid_%d" % (w*10), "model.w_con_loss="+str(w), "model.heatmap=add"]
        cfg = hydra.compose(config_name="conf", overrides=args)
        run = cluster.submit(args=args, title=cfg.exp_name)

# daug + hybrid-contrastive
with hydra.initialize(config_path='./configs', version_base=None):
    for w in [0.7, 0.8]:
        args = ["exp_name=daugadd_contra_%d" % (w*10), "model.w_con_loss="+str(w), "model.label_bce=False", "model.heatmap=add"]
        cfg = hydra.compose(config_name="conf", overrides=args)
        run = cluster.submit(args=args, title=cfg.exp_name)